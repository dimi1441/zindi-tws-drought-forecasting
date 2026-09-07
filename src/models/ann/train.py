"""Entraînement de l'ANN avec masquage augmenté par époque (Phase 5, demande utilisateur du
2026-09-06/07). `sklearn.neural_network.MLPRegressor` plutôt que PyTorch (décision du
2026-09-07, réseau bloqué sur l'installation de torch) : sa méthode `partial_fit()` fait
exactement ce qu'il faut -- une itération de plus à chaque appel, sur les données qu'on lui donne
-- donc pas besoin d'écrire de boucle de rétropropagation ni de masque de perte par ligne : on lui
donne directement les lignes de vrai train de cette époque.

Contrat identique à `baselines.fit_predict_lightgbm_early_stopping` : holdout à deux niveaux
(`fit_mask` jamais évalué pour la métrique finale, `inner_val_mask` sert uniquement à décider
quand arrêter). Contrairement au LSTM initialement envisagé, pas de `LayerNorm` (MLPRegressor ne
le propose pas) -- à la place, un `Pipeline(SimpleImputer(médiane) + StandardScaler)` est ajusté
UNE SEULE FOIS sur la vue réelle non masquée (jamais par époque, jamais sur la vue masquée),
même principe que la correction du 2026-09-06 sur `generate_submission.py` : tout ce qui doit
rester une référence stable (ici, l'échelle des features) s'ancre sur les vraies données, pas sur
une simulation de trous.
"""

import copy
from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.mask_augmentation import select_gap_months
from src.models.ann.dynamic_features import compute_dynamic_features
from src.models.ann.feature_tensors import FeatureScaffold, gather_static_features


def _assemble_feature_matrix(
    static_flat: dict[str, np.ndarray],
    dynamic_flat: dict[str, np.ndarray],
    feature_columns: list[str],
    row_mask: np.ndarray,
) -> np.ndarray:
    columns = [dynamic_flat.get(col, static_flat.get(col))[row_mask] for col in feature_columns]
    return np.column_stack(columns).astype(np.float32)


@dataclass
class TrainedANN:
    model: MLPRegressor
    preprocessor: Pipeline
    feature_columns: list[str]
    best_epoch: int
    best_inner_val_mae: float
    n_epochs_run: int


def train_ann(
    scaffold: FeatureScaffold,
    feature_columns: list[str],
    fit_mask: np.ndarray,
    inner_val_mask: np.ndarray,
    masking_config: dict,
    seed: int,
    hidden_layer_sizes: tuple[int, ...] = (128, 64, 32),
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    batch_size: int = 8192,
    max_epochs: int = 150,
    patience: int = 20,
    smoothing_window: int = 4,
    resample_mask_each_epoch: bool = True,
) -> TrainedANN:
    """Entraîne un `MLPRegressor` avec un tirage de masquage différent à chaque époque.

    `fit_mask`/`inner_val_mask` : masques booléens de longueur `len(scaffold.panel)`, disjoints,
    tous deux restreints aux lignes de train (jamais de lignes de test). Les features de
    `inner_val_mask` sont calculées UNE SEULE FOIS à partir de la vue réelle non masquée (jamais
    recalculées par époque) -- l'arrêt anticipé doit juger le modèle sur un scénario réaliste,
    pas sur une simulation de trous supplémentaire.

    `resample_mask_each_epoch` (diagnostic ajouté le 2026-09-07, premier test réel instable :
    meilleur epoch trouvé à l'époque 2 puis dégradation) : si `False`, un seul tirage de masquage
    est fait avant la boucle et réutilisé à toutes les époques (comme pour le GBR), au lieu d'un
    tirage différent à chaque époque. Sert à isoler si l'instabilité vient du changement de
    distribution d'entrée d'une époque à l'autre (qui perturbe les moments adaptatifs d'Adam) ou
    d'autre chose (architecture, prétraitement).
    """
    rng = np.random.default_rng(seed)
    static_flat = gather_static_features(scaffold)
    real_features = compute_dynamic_features(scaffold, set())  # vue réelle, jamais masquée
    y_all = scaffold.panel["target"].to_numpy(dtype=np.float32)

    X_fit_real = _assemble_feature_matrix(static_flat, real_features, feature_columns, fit_mask)
    preprocessor = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    preprocessor.fit(X_fit_real)

    X_inner_val = preprocessor.transform(
        _assemble_feature_matrix(static_flat, real_features, feature_columns, inner_val_mask)
    )
    y_inner_val = y_all[inner_val_mask]
    y_fit = y_all[fit_mask]

    existing_train_months = scaffold.panel.loc[scaffold.panel["is_train"], "time"]

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        batch_size=min(batch_size, int(fit_mask.sum())),
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=seed,
    )

    best_score = np.inf
    best_state: tuple | None = None
    best_epoch = -1
    patience_counter = 0
    recent_scores: list[float] = []
    epoch = 0

    fixed_X_fit = None
    if not resample_mask_each_epoch:
        fixed_gap_months = select_gap_months(
            existing_train_months, masking_config["target_gap_rate_by_period"], rng
        )
        fixed_features = compute_dynamic_features(scaffold, fixed_gap_months)
        fixed_X_fit = preprocessor.transform(
            _assemble_feature_matrix(static_flat, fixed_features, feature_columns, fit_mask)
        )

    for epoch in range(max_epochs):
        if resample_mask_each_epoch:
            gap_months = select_gap_months(
                existing_train_months, masking_config["target_gap_rate_by_period"], rng
            )
            epoch_features = compute_dynamic_features(scaffold, gap_months)
            X_fit = preprocessor.transform(
                _assemble_feature_matrix(static_flat, epoch_features, feature_columns, fit_mask)
            )
        else:
            X_fit = fixed_X_fit
        model.partial_fit(X_fit, y_fit)

        inner_mae = mean_absolute_error(y_inner_val, model.predict(X_inner_val))
        recent_scores.append(inner_mae)
        smoothed = float(np.mean(recent_scores[-smoothing_window:]))

        if smoothed < best_score - 1e-6:
            best_score = smoothed
            best_state = (copy.deepcopy(model.coefs_), copy.deepcopy(model.intercepts_))
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.coefs_, model.intercepts_ = best_state

    return TrainedANN(
        model=model,
        preprocessor=preprocessor,
        feature_columns=feature_columns,
        best_epoch=best_epoch,
        best_inner_val_mae=best_score,
        n_epochs_run=epoch + 1,
    )


def predict_ann(trained: TrainedANN, scaffold: FeatureScaffold, row_mask: np.ndarray) -> np.ndarray:
    """Prédit sur `row_mask` à partir de la vue RÉELLE non masquée (jamais un tirage augmenté --
    même principe que la correction du 2026-09-06 sur `generate_submission.py` : à l'inférence,
    l'historique réel et complet est disponible, le masquage augmenté n'a de sens qu'à
    l'entraînement)."""
    static_flat = gather_static_features(scaffold)
    real_features = compute_dynamic_features(scaffold, set())
    X = trained.preprocessor.transform(
        _assemble_feature_matrix(static_flat, real_features, trained.feature_columns, row_mask)
    )
    return trained.model.predict(X)
