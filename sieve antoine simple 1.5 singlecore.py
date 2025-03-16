import numpy as np
from bitarray import bitarray
from time import perf_counter as timer

t = timer()
limit = 100000000
factors_mod = [2, 3, 5, 7]  # must be primes and consecutive  from 2 to 13 more u have less memory it use but the corection array add become much more large over 13 os can handle it
mod_n = np.prod(factors_mod)
inv = np.array([i for i in range(1, mod_n) if np.gcd(i, mod_n) == 1], dtype=np.int64)
euler = inv.size
data = np.ones(int((limit ** 0.5)) // 2 + 1, dtype=np.bool_)
for i in range(1, int(limit ** 0.25) // 2 + 1):
    if data[i]:
        data[2 * i * (i + 1)::2 * i + 1] = False
primes = (np.nonzero(data)[0] * 2 + 1)[len(factors_mod):]
invidx = np.searchsorted(inv, primes % mod_n)           # primes reducted by mod   index of invertible
add = (inv[np.argsort((inv[:, None] * inv) % mod_n, axis=0)] - inv) % mod_n  # All combinations of the outer product (inv ⊗ inv), reduced by mod_n, adjusted by subtracting inv (mod mod_n), and sorted based on (inv ⊗ inv) reduced by mod_n.
print(f"\n  (\u2124/{mod_n}\u2124)ˣ={{ {','.join(map(str, inv[:3]))}...{','.join(map(str, inv[-3:]))} }}     φ({mod_n}) = {euler}\n")
print(f"  (\u2124/{mod_n}\u2124)ᶜ = {{x ∈ ℤ/{mod_n}ℤ | x divisible par {'*'.join(map(str, factors_mod))}}}\n\nℕ  {limit:.2e}")
init = timer()
data = {residue_idx: bitarray('1') * ((limit // mod_n) + 1) for residue_idx in range(euler)}
data[0][0] = False
for inv_idx in range(euler):
    for prime, first_factor_inv in zip(primes, primes * (primes + add[inv_idx][invidx]) // mod_n):
        data[inv_idx][first_factor_inv::prime] = 0b0
sieve = timer()
quotient, index = (np.nonzero(np.unpackbits(np.frombuffer(b"".join(dat.tobytes() for dat in data.values()), dtype=np.uint8)).reshape(len(inv), -1).T))
primes = np.concatenate([np.array(factors_mod)] + [(quotient * mod_n + inv[index])])
primes = primes[primes <= limit]
print(f"Init  :{init - t:.3f}\nSieve :{sieve - init:.3f}\nGen   :{(f := timer()) - sieve:.3f}\nTotal: {f - t:.3f}  Nb:{primes.size}")
