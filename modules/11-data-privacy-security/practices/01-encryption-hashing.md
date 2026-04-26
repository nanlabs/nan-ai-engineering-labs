# Practice 01 — Encryption and Hashing

## 🎯 Objectives

- Implement symmetric/asymmetric encryption
- Use hashing for passwords
- Secure key management
- Apply in ML pipelines

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: Hashing with bcrypt

```python
import bcrypt
import hashlib

# Password hashing
password = "my_secure_password".encode('utf-8')

# bcrypt (recommended for passwords)
salt = bcrypt.gensalt(rounds=12)
hashed = bcrypt.hashpw(password, salt)

print(f"Original: {password}")
print(f"Hashed: {hashed}")

# Verification
is_correct = bcrypt.checkpw(password, hashed)
print(f"Password correct: {is_correct}")

# SHA-256 para data integrity
data = "important data"
hash_obj = hashlib.sha256(data.encode())
data_hash = hash_obj.hexdigest()

print(f"\\nData hash (SHA-256): {data_hash}")

# Verificar integridad
received_data = "important data"
received_hash = hashlib.sha256(received_data.encode()).hexdigest()
print(f"Data integrity verified: {data_hash == received_hash}")
```

______________________________________________________________________

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: Fernet Encryption

**Statement:**
Encrypt datasets with Fernet:

```python
from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher = Fernet(key)
encrypted = cipher.encrypt(data)
```

### Exercise 2.2: RSA Asymmetric

**Statement:**
Implement RSA:

- Generates key pair (public/private)
- Encrypted with public
- Decrypt with private
- Firma digital

### Exercise 2.3: Homomorphic Encryption

**Statement:**
Usa PySEAL o Pyfhel:

- Opera about Data encriptados
- Addition/multiplication without decryption
- Aplica a Model lineal simple

### Exercise 2.4: Secure ML Pipeline

**Statement:**
Pipeline complete:

- Encripta features sensibles
- Train about encrypted features
- Desencripta Predictions
- Performance overhead measurement

### Exercise 2.5: Key Management

**Statement:**
Key management system:

- Key rotation
- Almacenamiento seguro (KeyVault simulation)
- Access logs
- Revocation

______________________________________________________________________

## ✅ Checklist

- [ ] Hashing with bcrypt and SHA-256
- [ ] Symmetric encryption (Fernet)
- [ ] Asymmetric encryption (RSA)
- [ ] Homomorphic encryption basic
- [ ] Secure key management

______________________________________________________________________

## 📚 Resources

- [Cryptography.io](https://cryptography.io/)
- [PyCryptodome Docs](https://pycryptodome.readthedocs.io/)
