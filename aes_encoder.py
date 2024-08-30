from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import binascii

def aes_decrypt(encrypted_message, key_hex, iv_hex):
    # Convert hex strings to bytes
    key = binascii.unhexlify(key_hex)
    iv = binascii.unhexlify(iv_hex)
    ciphertext = binascii.unhexlify(encrypted_message)
    
    # Initialize AES cipher
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    # Decrypt and unpad
    padded_plaintext = cipher.decrypt(ciphertext)
    plaintext = unpad(padded_plaintext, AES.block_size)
    
    return plaintext.decode('utf-8')

def get_input():
    # Get user input for key, IV, and encrypted message
    key_hex = input("Enter the key (hexadecimal): ")
    iv_hex = input("Enter the IV (hexadecimal): ")
    encrypted_message = input("Enter the encrypted message (hexadecimal): ")
    
    # Decrypt and output
    plaintext = aes_decrypt(encrypted_message, key_hex, iv_hex)
    print(f"Plaintext: {plaintext}")

if __name__ == "__main__":
    get_input()
