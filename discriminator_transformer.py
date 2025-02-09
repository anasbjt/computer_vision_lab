import tensorflow as tf
from tensorflow.keras import layers

def transformer_discriminator():
    """
    Fonction qui construit un discriminateur basé sur une architecture Transformer.
    Ce réseau prend une image en entrée et prédit si elle est réelle ou générée.
    """
    
    # Définition de l'entrée du modèle : une image 28x28 avec 1 canal (niveau de gris)
    inputs = layers.Input(shape=(28, 28, 1))  
    
    # Aplatissement de l'image en un vecteur 1D
    x = layers.Flatten()(inputs)  
    
    # Première couche dense avec activation ReLU
    x = layers.Dense(128, activation="relu")(x)  
    # - Convertit l'entrée en un vecteur de 128 dimensions
    # - ReLU permet d'apprendre des représentations complexes

    # Reshape pour organiser les données en "patches" (blocs d'informations)
    x = layers.Reshape((49, 128))(x)  
    # - 49 patches (7x7) avec 128 caractéristiques chacun
    # - Cette organisation est nécessaire pour l'attention multi-tête
    
    # Création de l'encodage de position pour donner un sens aux patches dans l'espace
    position_encoding = tf.range(start=0, limit=49, delta=1)  
    # - Génère un vecteur de positions [0, 1, 2, ..., 48] correspondant aux 49 patches

    # Ajout d'un embedding positionnel
    position_embedding = layers.Embedding(input_dim=49, output_dim=128)(position_encoding)  
    # - Associe une représentation apprise à chaque position pour que le modèle comprenne l'ordre spatial
    
    # Ajout de l'encodage positionnel aux données (x)
    x += position_embedding  
    # - Permet au modèle de différencier les patches entre eux en ajoutant de l'information spatiale
    
    # Multi-Head Self-Attention pour capturer les relations globales entre les patches
    x = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x, x)  
    # - 4 têtes d'attention permettent d'apprendre différentes relations entre patches
    # - La clé (key_dim) est de 128, correspondant aux dimensions des caractéristiques
    
    # Normalisation des valeurs pour stabiliser l'entraînement
    x = layers.LayerNormalization()(x)  
    
    # Aplatissement des données après l'attention
    x = layers.Flatten()(x)  
    
    # Couche dense pour la classification avec activation ReLU
    x = layers.Dense(128, activation="relu")(x)  
    
    # Couche de sortie avec activation sigmoïde pour renvoyer une probabilité entre 0 et 1
    x = layers.Dense(1, activation="sigmoid")(x)  
    # - Donne une probabilité indiquant si l'image est réelle ou générée
    
    # Création du modèle
    model = tf.keras.Model(inputs, x, name="Transformer_Discriminator")  
    
    return model
