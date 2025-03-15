from multiprocessing import Pool
from bitarray import bitarray
import numpy as np
from time import perf_counter as timer


def generate_splited(is_2last, limit, totatives, mod, offset, parts):
    data = (np.concatenate([(np.nonzero(np.unpackbits(np.frombuffer(part.tobytes(), dtype=np.uint8)))[0]
                            + offset) * mod + totatives[i] for i, part in enumerate(parts)]))
    data.sort()
    return data if not is_2last else data[data <= limit]


def erathosthene(limit):
    data = np.ones(limit // 2 + 1, dtype=np.bool_)
    for i in range(1, int(limit ** 0.5) // 2 + 1):
        if data[i]:
            data[2 * i * (i + 1)::2 * i + 1] = False
    primes = np.nonzero(data)[0] * 2 + 1
    primes[0] = 2
    return primes


def by_totative(size, mod, primes, idx, row):
    data = bitarray('1') * size
    for prime, sqr_in_residue in zip(primes, primes * (primes + row[idx]) // mod):
        data[sqr_in_residue::prime] = 0b0
    return data


def sieve_totatives_multicore(limit, div, parts, core):
    t = timer()
    mod_n = int(np.prod(div))
    inv = np.array([i for i in range(1, mod_n) if np.gcd(i, mod_n) == 1], dtype=np.int64)
    euler_totient = inv.size
    primes = erathosthene(int(limit ** 0.5) + 1)[len(div):] # list of primes
    invidx = np.searchsorted(inv, primes % mod_n)           # same but reducted by mod  giving invertible  storing index of them
    add = (inv[np.argsort((inv[:, None] * inv) % mod_n, axis=0)] - inv)% mod_n  # all combinaison of inv * inv % mod reordered and do - inv % mod, stored by list of invertible
    offset = [(limit // mod_n + 1) // parts * i for i in range(parts)] + [limit // mod_n + 1]  # list of slicer
    with Pool(core) as pool:
        print((f"\n Erathostene v 1.5 optimized Coprimes (\u2124/{mod_n}\u2124)ˣ = {{ {','.join(map(str, inv[:3]))}..."
               f"{','.join(map(str, inv[-3:]))} }} \n                              Excluded (\u2124/{mod_n}\u2124)ᶜ = "
               f"{{x ∈ ℤ/{mod_n}ℤ | x divisible par {'*'.join(map(str, div))}}}\nℕ  {limit:.2e}"))
        init = timer()
        results = pool.starmap(by_totative, [(offset[-1], mod_n, primes, invidx, add[i]) for i in range(euler_totient)])
        results[0][0] = 0
        sieve = timer()
        args = [(i >= parts - 2, limit, inv, mod_n, offset[i],
                 [results[j][offset[i]:offset[i + 1]] for j in range(euler_totient)]) for i in range(parts)]
        primes = np.concatenate([np.array(div)] + pool.starmap(generate_splited, args))  # primes is a list of all primes
        print(f"init  :{init - t:.3f}\nSieve :{sieve-init:.3f}\nGen   :{(f := timer())-sieve:.3f}\nTotal :{f - t:.3f}")
        print(len(primes)) 


if __name__ == "__main__":
    sieve_totatives_multicore(limit=10000000000, div=[2, 3, 5, 7, 11], parts=46, core=12)
