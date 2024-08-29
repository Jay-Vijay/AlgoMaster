import pandas as pd
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from Crypto.Cipher import ChaCha20 as ChaCha20Cipher
import os
import random
import string

# Function to encrypt a message using a given algorithm
def encrypt_message(algorithm, key, iv, plaintext):
    if algorithm == 'AES':
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    elif algorithm == 'DES' or algorithm == '3DES':
        cipher = Cipher(algorithms.TripleDES(key), modes.CBC(iv), backend=default_backend())
    elif algorithm == 'Blowfish':
        cipher = Cipher(algorithms.Blowfish(key), modes.CBC(iv), backend=default_backend())
    elif algorithm == 'RC4':
        cipher = Cipher(algorithms.ARC4(key), mode=None, backend=default_backend())
    elif algorithm == 'ChaCha20':
        cipher = ChaCha20Cipher.new(key=key, nonce=iv)
    else:
        raise ValueError("Unsupported algorithm")

    if algorithm in ['AES', 'DES', '3DES', 'Blowfish']:
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(plaintext.encode()) + padder.finalize()
        encrypted_message = encryptor.update(padded_data) + encryptor.finalize()
    elif algorithm == 'RC4' or algorithm == 'ChaCha20':
        encrypted_message = cipher.encrypt(plaintext.encode())
    else:
        raise ValueError("Unsupported algorithm")

    return encrypted_message

# Generate random key and IV for each algorithm
def generate_key_iv(algorithm):
    if algorithm == 'AES':
        key = os.urandom(32)  # AES-256
        iv = os.urandom(16)
    elif algorithm == 'DES' or algorithm == '3DES':
        key = os.urandom(24)  # 3DES
        iv = os.urandom(8)
    elif algorithm == 'Blowfish':
        key = os.urandom(16)  # Blowfish
        iv = os.urandom(8)
    elif algorithm == 'RC4':
        key = os.urandom(16)  # RC4
        iv = None
    elif algorithm == 'ChaCha20':
        key = os.urandom(32)  # ChaCha20
        iv = os.urandom(8)
    else:
        raise ValueError("Unsupported algorithm")
    return key, iv

# Function to generate random plaintext messages with varying lengths
def generate_random_plaintext(min_length=10, max_length=100):
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=length))

# Function to generate meaningful plaintext messages
def generate_meaningful_plaintext():
    templates = [
        "This is a {adjective} message about {topic}.",
        "The {subject} is very important for {topic}.",
        "In the world of {topic}, {subject} plays a crucial role.",
        "Understanding {topic} requires knowledge of {subject}.",
        "The {adjective} {subject} is essential for {topic}.",
        "Learning about {topic} can be {adjective} but rewarding.",
        "The {subject} is a key component of {topic}.",
        "Exploring {topic} reveals the importance of {subject}.",
        "The {adjective} nature of {subject} is evident in {topic}.",
        "In the context of {topic}, {subject} is {adjective}."
    ]

    adjectives = [
        "important", "interesting", "complex", "simple", "critical", "fascinating", "challenging", "essential", "vital",
        "innovative", "creative", "dynamic", "efficient", "effective", "productive", "strategic", "tactical", "operational",
        "collaborative", "cooperative", "inclusive", "diverse", "global", "international", "regional", "local", "national"
    ]

    subjects = [
        "encryption", "security", "data integrity", "confidentiality", "authentication", "cryptography", "algorithm", "key management",
        "environmental conservation", "climate change", "renewable energy", "sustainable development",
        "global health", "pandemic response", "public health", "medical research",
        "international relations", "global politics", "diplomacy", "peacekeeping",
        "economic development", "financial markets", "trade agreements", "economic policy",
        "social justice", "human rights", "equality", "social welfare",
        "education reform", "higher education", "lifelong learning", "educational technology",
        "artificial intelligence", "machine learning", "robotics", "automation",
        "space exploration", "astronomy", "astrophysics", "space technology",
        "cultural heritage", "art history", "literature", "performing arts",
        "innovation", "entrepreneurship", "startups", "business development",
        "community engagement", "civic participation", "volunteerism", "philanthropy"
    ]

    topics = [
        "cybersecurity", "information technology", "data protection", "network security", "digital forensics", "computer science",
        "environmental conservation", "climate change", "renewable energy", "sustainable development",
        "global health", "pandemic response", "public health", "medical research",
        "international relations", "global politics", "diplomacy", "peacekeeping",
        "economic development", "financial markets", "trade agreements", "economic policy",
        "social justice", "human rights", "equality", "social welfare",
        "education reform", "higher education", "lifelong learning", "educational technology",
        "artificial intelligence", "machine learning", "robotics", "automation",
        "space exploration", "astronomy", "astrophysics", "space technology",
        "cultural heritage", "art history", "literature", "performing arts",
        "innovation", "entrepreneurship", "startups", "business development",
        "community engagement", "civic participation", "volunteerism", "philanthropy"
    ]

    template = random.choice(templates)
    message = template.format(
        adjective=random.choice(adjectives),
        subject=random.choice(subjects),
        topic=random.choice(topics)
    )
    return message

# List of algorithms to use
algorithms_list = ['AES', 'DES', '3DES', 'Blowfish', 'RC4', 'ChaCha20']

# Number of plaintext messages to generate for each algorithm
num_messages = 300

# Create the dataset
dataset = []

for algorithm in algorithms_list:
    for _ in range(num_messages):
        key, iv = generate_key_iv(algorithm)

        # Randomly choose between a meaningful message and a random message
        if random.choice([True, False]):
            plaintext = generate_meaningful_plaintext()
        else:
            plaintext = generate_random_plaintext()

        try:
            encrypted_message = encrypt_message(algorithm, key, iv, plaintext)
            dataset.append({
                'algorithm': algorithm,
                'plaintext': plaintext,
                'key': key.hex(),
                'iv': iv.hex() if iv else None,
                'encrypted_message': encrypted_message.hex()
            })
        except Exception as e:
            print(f"Error encrypting with {algorithm} in CBC mode: {e}")

# Convert the dataset to a DataFrame
df = pd.DataFrame(dataset)

# Save the dataset to a CSV file
output_path = 'C:/Users/priya/Documents/SIH_JAYY/datasets/encrypted_messages_dataset12.csv'
df.to_csv(output_path, index=False)

print(f"Dataset created and saved to '{output_path}'")
