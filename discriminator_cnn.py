import tensorflow as tf
from tensorflow.keras import layers

def build_discriminator():
    """
    Fonction qui construit le discriminateur d'un GAN.
    Le discriminateur est un réseau de neurones convolutionnel qui prend une image en entrée
    et détermine si elle est réelle ou générée par le générateur.
    """
    model = tf.keras.Sequential([  # Création d'un modèle séquentiel où les couches sont empilées les unes après les autres
        
        # Première couche de convolution
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=(28, 28, 1)),
        # - 64 filtres de convolution pour extraire des caractéristiques
        # - Taille du noyau de 4x4
        # - Stride de 2 (réduction de la taille de l'image par 2)
        # - Padding "same" pour conserver la taille après la convolution
        # - Input_shape (28, 28, 1) signifie que le modèle attend une image 28x28 en niveaux de gris (1 canal)

        layers.LeakyReLU(alpha=0.2),  # Fonction d'activation LeakyReLU avec un paramètre alpha de 0.2
        # - Permet d'éviter le problème des neurones morts (dying ReLU)
        # - Les valeurs négatives sont multipliées par 0.2 au lieu d'être mises à zéro

        # Deuxième couche de convolution
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        # - 128 filtres de convolution pour capturer des caractéristiques plus complexes
        # - Taille du noyau de 4x4
        # - Stride de 2 (réduction encore de la taille de l'image)
        # - Padding "same" pour maintenir les dimensions après convolution

        layers.LeakyReLU(alpha=0.2),  # Activation LeakyReLU encore pour éviter le problème des valeurs nulles
        
        layers.Flatten(),  # Aplatissement des matrices 2D en un vecteur 1D
        # - Convertit les caractéristiques extraites en un vecteur pour la classification

        layers.Dense(1, activation="sigmoid")  # Couche de sortie avec une seule neurone et activation sigmoïde
        # - Produit une valeur entre 0 et 1, interprétée comme la probabilité que l'image soit réelle (1) ou générée (0)
    ])
    
    return model
