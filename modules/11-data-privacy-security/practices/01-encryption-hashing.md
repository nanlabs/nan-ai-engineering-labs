# Práctica 01 — Encriptación y Hashing

## 🎯 Objetivos

- Implementar encriptación simétrica/asimétrica
- Usar hashing para passwords
- Secure key management
- Aplicar en ML pipelines

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Hashing con bcrypt

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

# Verificación
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

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Fernet Encryption

**Enunciado:**
Encripta datasets con Fernet:

```python
from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher = Fernet(key)
encrypted = cipher.encrypt(data)
```

### Ejercicio 2.2: RSA Asymmetric

**Enunciado:**
Implementa RSA:

- Genera par de llaves (pública/privada)
- Encripta con pública
- Desencripta con privada
- Firma digital

### Ejercicio 2.3: Homomorphic Encryption

**Enunciado:**
Usa PySEAL o Pyfhel:

- Opera sobre datos encriptados
- Suma/multiplicación sin desencriptar
- Aplica a modelo lineal simple

### Ejercicio 2.4: Secure ML Pipeline

**Enunciado:**
Pipeline completo:

- Encripta features sensibles
- Entrena sobre encrypted features
- Desencripta predicciones
- Mide overhead de performance

### Ejercicio 2.5: Key Management

**Enunciado:**
Sistema de gestión de llaves:

- Key rotation
- Almacenamiento seguro (KeyVault simulation)
- Access logs
- Revocation

______________________________________________________________________

## ✅ Checklist

- [ ] Hashing con bcrypt y SHA-256
- [ ] Encriptación simétrica (Fernet)
- [ ] Encriptación asimétrica (RSA)
- [ ] Homomorphic encryption básico
- [ ] Secure key management

______________________________________________________________________

## 📚 Recursos

- [Cryptography.io](https://cryptography.io/)
- [PyCryptodome Docs](https://pycryptodome.readthedocs.io/)
