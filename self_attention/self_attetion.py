import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
# Define a small vocabulary and a sequence
vocab = {"I": 0, "love": 1, "machine": 2, "learning": 3}
sequence = [0, 1, 2, 3]  # Corresponding to "I love machine learning"
 
# Self-Attention Class
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.output_dense = tf.keras.layers.Dense(embed_dim)
    
    def call(self, inputs):
        Q = self.query_dense(inputs)
        K = self.key_dense(inputs)
        V = self.value_dense(inputs)
        
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attention_weights, V)
        return output, attention_weights
 
# Embedding layer
embedding_dim = 8
num_heads = 2
embedding_layer = tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=embedding_dim)
embedded_sequence = embedding_layer(tf.convert_to_tensor(sequence))
 
# Apply Self-Attention
self_attention = SelfAttention(embed_dim=embedding_dim, num_heads=num_heads)
attention_output, attention_weights = self_attention(embedded_sequence)
 
# Visualize Attention Scores
sns.heatmap(attention_weights.numpy(), annot=True, cmap='coolwarm', xticklabels=vocab.keys(), yticklabels=vocab.keys())
plt.title("Attention Scores")
plt.show()