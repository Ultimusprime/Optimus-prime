from multiprocessing import Pool
from bitarray import bitarray
import numpy as np
from time import perf_counter


class Chrono:
    """Chrono class for measuring elapsed time.

    **example:**

    Initialize instance with actual time

    * bench = ``Chrono()``

    Add entries, evaluate the time, and return formatted text.
    -- params default (3, False)

    * print(f"Bench: {``bench.mark()``}")
    :NOTE:
    This class uses *perf_counter* function from the *time* module for time measurements."""

    def __init__(self):
        """Initialize instance with actual time"""

        self.bench = [perf_counter()]

    def mark(self, precision=3, total=False):
        """Elapsed time since previous use

        :arg int precision: + precision to be formated.
        :arg bool total: + add since begining or not.
        :return: formated text.
        :rtype: str"""

        self.bench.append(perf_counter())
        return f" {self.bench[-1] - self.bench[-2]:.{precision}f} sec.  {"" if not total else f"total: {self.bench[-1] - self.bench[0]:.{precision}f} sec."} "


def sieve_antoine(n: int, exclude: list):
    mod = np.prod(exclude)
    totatives = [i for i in range(1, mod) if np.gcd(i, mod) == 1]
    data = {i: bitarray('1') * (n // mod + 1) for i in totatives}
    for nth in range(0, int(n ** 0.5) // (mod - 1) + 1):
        for i in totatives:
            if nth == 0 and i == 1:
                continue
            if data[i][nth]:
                step = nth * mod + i
                for x in totatives:
                    if x < i:
                        x += mod
                    start_index = (step ** 2 + step * (x - i)) // mod
                    data[(x * i) % mod][start_index::step] = 0b0
    data[1][0] = False  # remove 1 as prime
    for i in totatives:  # sieve is a modulus factor, the last module needs correction to fit within the limit
        if data[i][-1]:
            last_index = (n // mod) * mod + i
            if last_index > n:  # remove primes beyond the actual limit
                data[i][-1] = False
    lst = [nth * mod + i for i in totatives for nth in np.nonzero(np.frombuffer(data[i].unpack(), dtype=np.uint8))[0]]
    lst.sort()
    return list(exclude) + lst


def sieve_totative(size, mod, master_loop, totatives, add):
    """
    Apply the sieve of Eratosthenes variant using totative to mark non-prime indices in a bitarray.

    :param size: The size of the array to sieve.
    :param mod: Modulus value
    :param master_loop: List of lists where each sublist contains indices corresponding to indices in totatives. (primes under limite**0.5 +1).
    :param totatives: Totatives list.
    :param add: List of offsets to adjust each totative.
    :return: A bitarray where prime indices are still at True and non-prime indices are marked as False.
    """
    # Initialisation de bitarray
    data = bitarray('1') * size
    # rebuild the other half of add table
    add = np.r_[add, (mod - np.flip(add)) % mod].tolist()
    #  level of sublist in master loop
    for nth in range(len(master_loop)):
        #  Each sublist in master_loop corresponds to indices of totatives for prime numbers under modulus.
        for ix in master_loop[nth]:
            # convert data into real number
            step = nth * mod + totatives[ix]
            # mark non primes
            data[(step ** 2 + step * add[ix]) // mod::step] = 0b0
    return data


def generate_splited(limit, is_last, totatives, mod, offset, parts):
    """
    Generate and split data based on provided parameters and return sorted values.

    :param int limit: Upper limit of sieves.
    :param bool is_last: Last part flag.
    :param list totatives: Totatives list.
    :param int mod: Modulus value
    :param int offset: The offset value on the y-axis.
    :param list[bitarray] parts: Data parts along the y-axis.
    :return :Sorted array of generated data values up to limit.
    :rtype: numpy.ndarray
    """

    # Unpack bitarray, adjust indices by offset, * by mod, + the respective totative into list, and flatten all lists .
    data = np.concatenate([(np.nonzero(np.frombuffer(parts[_].unpack(), dtype=np.int8))[0] + offset) * mod + totatives[_] for _ in range(len(parts))])
    data.sort()
    return data if not is_last else data[data <= limit]


def sieve_totatives_multicore(limit, exclude, parts, core):
    """
    Multi-core sieve algorithm to find primes sieving each totative.

    :param int limit: The upper limit of numbers to sieve.
    :param list exclude: List of numbers to exclude (used to calculate the modulus).
    :param int parts: Number of parts to divide the generated work into.
    :param int core: Number of cores to use for multiprocessing.
    :return: Flattened lists of primes.
    :rtype: np.ndarray
    """
    # initilisation
    mod = int(np.prod(exclude))
    nb_tot = len(totatives := [i for i in range(1, mod) if np.gcd(i, mod) == 1])
    offset = [(limit // mod + 1) // parts * i for i in range(parts)] + [limit // mod + 1]

    # primes list up to (limit ** 0.5) + 1
    j = sieve_antoine(int(limit ** 0.5) + 1, [2, 3, 5])[len(exclude):]

    # format list into sublist level of totatives index
    master_loop = [[] for _ in range(j[-1] // mod + 1)]
    [master_loop[i // mod].append(totatives.index(i % mod)) for i in j]

    # half of precalculated table, transpose (for sieving by totative)
    add = np.zeros((nb_tot, nb_tot // 2), np.int64)
    for i in range(nb_tot // 2):
        rolled_totatives = np.roll(totatives, -i)
        sorted_indices = np.argsort(rolled_totatives * totatives[i] % mod)
        add[:, i] = ((rolled_totatives[sorted_indices] - totatives[i]) % mod)
    print(f"\nLimite: {limit:.4e} Mod {mod} Residus de classe: {nb_tot}")
    print(f"initialisation:        {bench.mark()}")

    # multicore
    with Pool(core) as pool:
        # sieve
        results = pool.starmap(sieve_totative, [(offset[-1], mod, master_loop, totatives, add[i]) for i in range(nb_tot)])
        results[0][0] = 0
        print(f"sieve:                  {bench.mark()}")

        # generate
        args = [(limit, i == parts - 1,  totatives, mod, offset[i],  [results[j][offset[i]:offset[i+1]] for j in range(nb_tot)]) for i in range(parts)]
        results = pool.starmap(generate_splited, args)

    return np.concatenate([np.array(exclude)] + results)


if __name__ == "__main__":
    bench = Chrono()
    primes = sieve_totatives_multicore(limit=10000000000, exclude=[2, 3, 5, 7, 11], parts=46, core=12)
    print(f"generating primes: {bench.mark(total=True)} {primes.size}")


# 2 result on i9 10920x
# Limite: 1.0000e+10 Mod 2310 Residus de classe: 480
# initialization:         0.079 sec.   
# sieve:                   0.811 sec.   
# generating primes in list:  5.844 sec.  total: 6.733 sec.  455052511

# Limite: 1.0000e+11 Mod 2310 Residus de classe: 480
# initialisation:         0.115 sec.   
# crible:                   21.601 sec.   
# generating primes in list:  60.099 sec.  total: 81.816 sec.  4118054813
