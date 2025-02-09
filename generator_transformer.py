import tensorflow as tf  
from tensorflow.keras import layers  

def transformer_generator(latent_dim=100):
    """
    Fonction qui construit un générateur basé sur une architecture Transformer.
    Ce modèle prend un vecteur aléatoire de dimension `latent_dim` et génère une image réaliste.

    Arguments :
    latent_dim -- La dimension du vecteur latent (par défaut 100)

    Retourne :
    Un modèle Keras générateur basé sur l'attention.
    """

    # Définition de l'entrée : un vecteur latent de dimension `latent_dim`
    inputs = layers.Input(shape=(latent_dim,))

    # Couche Dense pour projeter le vecteur latent dans un espace de taille 7x7x128
    x = layers.Dense(7 * 7 * 128, activation="relu")(inputs)
    # - Convertit le vecteur latent en un tenseur de taille 7x7 avec 128 canaux
    # - Activation ReLU pour introduire de la non-linéarité

    # Reshape en 49 patches de 128 dimensions chacun (7x7 patches)
    x = layers.Reshape((49, 128))(x)
    # - Transforme les valeurs obtenues en une séquence de 49 "patches" de 128 dimensions

    # Création des encodages positionnels pour injecter une information spatiale
    position_encoding = tf.range(start=0, limit=49, delta=1)
    # - Génère un vecteur contenant les indices de position des patches [0, 1, ..., 48]

    position_embedding = layers.Embedding(input_dim=49, output_dim=128)(position_encoding)
    # - Associe un vecteur d'embedding de 128 dimensions à chaque position
    # - Permet au modèle de comprendre la structure spatiale de l'image

    x += position_embedding
    # - Ajout des embeddings de position aux patches (permet d'intégrer l'ordre spatial)

    # Application de l'attention multi-tête sur les patches
    x = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
    # - 4 têtes d'attention permettent de capturer différentes relations spatiales entre les patches
    # - `key_dim=128` correspond à la dimension des caractéristiques utilisées pour l'attention

    # Normalisation des valeurs pour stabiliser l'entraînement
    x = layers.LayerNormalization()(x)

    # Ajout d'une couche Dense supplémentaire pour affiner les caractéristiques extraites
    x = layers.Dense(128, activation="relu")(x)

    # Normalisation des valeurs après transformation
    x = layers.LayerNormalization()(x)

    # Reshape pour revenir à une image de taille 7x7 avec 128 canaux
    x = layers.Reshape((7, 7, 128))(x)

    # Convolution transposée pour augmenter la résolution à 14x14
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu")(x)
    # - 64 filtres de convolution transposée pour doubler la taille de l'image
    # - Stride=2 pour passer de 7x7 à 14x14
    # - Activation ReLU pour apprendre des représentations complexes

    # Convolution transposée finale pour obtenir une image 28x28 avec un seul canal
    x = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="tanh")(x)
    # - 1 seul canal en sortie (image en niveaux de gris)
    # - Activation tanh pour que les pixels soient dans l'intervalle [-1, 1]

    # Création du modèle final
    model = tf.keras.Model(inputs, x, name="Transformer_Generator")

    return model  
