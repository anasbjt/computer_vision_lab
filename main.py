import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import requests  # Importation de requests pour télécharger les données

# Importation des modules contenant les générateurs et discriminateurs des GANs
import generator_cnn, discriminator_cnn, generator_transformer, discriminator_transformer

# Téléchargement du dataset MNIST à partir du lien officiel de TensorFlow
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
response = requests.get(url)  # Envoi de la requête pour récupérer le fichier
with open("mnist.npz", "wb") as f:  # Ouverture d'un fichier en mode écriture binaire
    f.write(response.content)  # Sauvegarde des données téléchargées

# Chargement du dataset MNIST
data = np.load("mnist.npz")  # Chargement du fichier téléchargé
x_train, x_test = data["x_train"], data["x_test"]  # Extraction des ensembles d'entraînement et de test

# Prétraitement des données
x_train = x_train.astype("float32") / 255.0  # Normalisation des valeurs des pixels entre 0 et 1
x_train = np.expand_dims(x_train, -1)  # Ajout d'une dimension supplémentaire pour correspondre aux formats d'entrée de TensorFlow

# Définition de la dimension du vecteur latent (bruit aléatoire d'entrée du générateur)
latent_dim = 100  

# Création du modèle GAN
generator = generator_cnn.build_generator()  # Instanciation du générateur basé sur CNN
discriminator = discriminator_cnn.build_discriminator()  # Instanciation du discriminateur basé sur CNN

# Compilation du discriminateur
discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(0.0002),  # Optimiseur Adam avec un taux d'apprentissage de 0.0002
    loss="binary_crossentropy",  # Fonction de perte pour la classification binaire (réel vs. généré)
    metrics=["accuracy"]  # Suivi de l'exactitude du discriminateur
)
discriminator.trainable = False  # Désactivation de l'entraînement du discriminateur pendant l'entraînement du GAN

# Création du modèle GAN
gan_input = layers.Input(shape=(latent_dim,))  # Définition de l'entrée du GAN (vecteur latent)
gan_output = discriminator(generator(gan_input))  # Passage du vecteur latent dans le générateur, puis dans le discriminateur
gan = tf.keras.Model(gan_input, gan_output)  # Création du modèle GAN combinant le générateur et le discriminateur

# Compilation du GAN
gan.compile(
    optimizer=tf.keras.optimizers.Adam(0.0002),  # Utilisation d'Adam pour l'optimisation
    loss="binary_crossentropy"  # La perte utilisée pour entraîner le générateur
)

# Fonction d'entraînement du GAN
def train_gan(generator, discriminator, gan, epochs, batch_size=128):
    """
    Fonction d'entraînement du GAN.
    
    Arguments :
    generator -- Le générateur du GAN
    discriminator -- Le discriminateur du GAN
    gan -- Le modèle complet du GAN
    epochs -- Nombre d'époques d'entraînement
    batch_size -- Taille des mini-lots utilisés pour l'entraînement
    """
    for epoch in range(epochs):  # Boucle sur le nombre d'époques
        for _ in range(batch_size):  # Boucle sur le nombre d'échantillons par batch
            
            # Génération d'un batch d'images artificielles
            noise = tf.random.normal([batch_size, latent_dim])  # Génération de bruit aléatoire
            fake_images = generator.predict(noise)  # Génération d'images à partir du bruit

            # Sélection aléatoire d'images réelles
            real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]

            # Création des labels pour l'entraînement du discriminateur
            real_labels = tf.ones((batch_size, 1))  # Labels pour les images réelles (1)
            fake_labels = tf.zeros((batch_size, 1))  # Labels pour les images générées (0)

            # Entraînement du discriminateur
            d_loss_real = discriminator.train_on_batch(real_images, real_labels)  # Apprentissage sur les images réelles
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)  # Apprentissage sur les images générées

            # Entraînement du générateur via le modèle GAN
            misleading_labels = tf.ones((batch_size, 1))  # On veut tromper le discriminateur, donc on met 1
            g_loss = gan.train_on_batch(noise, misleading_labels)  # Mise à jour des poids du générateur

        # Affichage des pertes du générateur et du discriminateur
        print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss_real[0] + d_loss_fake[0]}, G Loss: {g_loss}")

# Lancement de l'entraînement du GAN
train_gan(generator, discriminator, gan, epochs=50)

# Fonction pour générer et visualiser des images
def generate_images(generator, n_images):
    """
    Fonction pour générer et afficher des images à partir du générateur.

    Arguments :
    generator -- Le générateur entraîné
    n_images -- Nombre d'images à générer et afficher
    """
    noise = tf.random.normal([n_images, latent_dim])  # Génération d'un vecteur de bruit aléatoire
    generated_images = generator.predict(noise)  # Génération des images
    
    # Création d'une figure pour afficher les images générées
    fig, axes = plt.subplots(1, n_images, figsize=(20, 4))
    for i, img in enumerate(generated_images):  # Boucle sur les images générées
        axes[i].imshow(img.squeeze(), cmap="gray")  # Affichage de l'image en niveaux de gris
        axes[i].axis("off")  # Suppression des axes pour une meilleure visibilité

    plt.show()  # Affichage des images générées

# Génération et affichage de 10 images à partir du générateur entraîné
generate_images(generator, 10)
