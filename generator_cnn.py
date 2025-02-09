import tensorflow as tf
from tensorflow.keras import layers

def build_generator(latent_dim=100):
    """
    Fonction qui construit le générateur d'un GAN.
    Ce modèle prend un vecteur de bruit aléatoire et le transforme en une image réaliste.
    
    Arguments :
    latent_dim -- La dimension du vecteur latent d'entrée (par défaut 100)

    Retourne :
    Un modèle Keras générateur
    """
    
    model = tf.keras.Sequential([  # Création d'un modèle séquentiel où les couches sont empilées les unes après les autres
        
        # Couche Dense pour transformer le vecteur latent en une matrice 7x7 avec 256 canaux
        layers.Dense(7 * 7 * 256, input_dim=latent_dim),
        # - Prend un vecteur de taille `latent_dim` (ex: 100)
        # - Le projette en une matrice de taille 7*7*256 (soit 12544 valeurs)
        
        layers.Reshape((7, 7, 256)),  
        # - Reformate le tenseur de sortie en une image de 7x7 avec 256 canaux (profondeur)
        # - Cela sert de base pour construire une image de résolution plus grande
        
        # Première couche de convolution transposée pour agrandir l'image à 14x14
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"),
        # - 128 filtres pour extraire des caractéristiques
        # - Taille du noyau 4x4
        # - Stride de 2 pour doubler la taille de l'image (passage de 7x7 à 14x14)
        # - Activation ReLU pour introduire de la non-linéarité
        
        # Deuxième couche de convolution transposée pour agrandir à 28x28
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"),
        # - 64 filtres
        # - Taille du noyau 4x4
        # - Stride de 2 pour doubler la taille (passage de 14x14 à 28x28)
        # - Activation ReLU

        # Dernière couche de convolution transposée pour générer une image finale 28x28 avec 1 canal
        layers.Conv2DTranspose(1, kernel_size=7, activation="tanh", padding="same"),
        # - 1 seul filtre car l'image de sortie est en niveaux de gris (1 canal)
        # - Taille du noyau 7x7
        # - Activation tanh : nécessaire pour générer des valeurs comprises entre -1 et 1,
        #   car les images MNIST sont normalisées entre -1 et 1
    ])
    
    return model  
