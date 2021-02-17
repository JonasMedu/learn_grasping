from hand_env.allegro_env import AllegroHand
from hand_env.allegro_env_unsupervised import AllegroCollisionFromTactileHand, AllegroCollisionPositionHand
from hand_env.noisy_hand import AllegronNoiseFromUpper
from hand_env.trained_env import TrainedEnv

allegro_gyms = {
    AllegroHand.__name__: AllegroHand,
    AllegroCollisionFromTactileHand.__name__: AllegroCollisionFromTactileHand,
    AllegroCollisionPositionHand.__name__: AllegroCollisionPositionHand,
    AllegronNoiseFromUpper.__name__: AllegronNoiseFromUpper,
    TrainedEnv.__name__: TrainedEnv

}