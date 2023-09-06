from bitarray import bitarray
from multiprocessing import Pool
import pickle
import os
from time import perf_counter as timer


class Chrono:
    """Class to measure time intervals and store timing information."""
    def __init__(self):
        """Get current time in a list."""
        self.bench = [timer()]

    def mark(self, tot=False, precision=2, layout="min"):
        """Mark a new event and return the time elapsed since the begining or previous marker.
        Args:
            tot (bool): Default (False) result with previous marker. (True) with the first marker.
            precision (int): Default (2), seconds precision.
            layout (str): Default ("min") display min or ("sec") seconds only.
        Returns:
            str: Time elapsed in the specified format."""
        self.bench.append(timer())
        seconds_elapsed = self.bench[-1] - self.bench[-2] if not tot else self.bench[-1] - self.bench[0]
        return f"{seconds_elapsed:.{precision}f} sec." if layout == "sec" else f"{seconds_elapsed // 60:.0f} min. {seconds_elapsed % 60:.{precision}f} sec."


def sieve_eratosthene(n, verbose=True):
    primes = bitarray()
    primes.extend(bitarray('1') * (n >> 1))
    print(f"\033[92mMemory use: {n/2/1014/1024:.2f} MB\033[0m by Eratosthene") if verbose else None

    for i in range(1, (int(n ** 0.5) + 1 >> 1) + 1):
        if primes[i]:
            p_sqr_div2 = (p := (i << 1) + 1) ** 2 >> 1
            primes[p_sqr_div2::p] = False
    print(f"Sieve core:       {bench.mark()}") if verbose else None

    yield 2
    for _ in range(1, len(primes)):
        if primes[_]:
            yield (_ << 1) + 1


class Constants:
    """Constant values used in sieve calculations.

    Attributes:
        limit (int): The limit of the sieve.
        modulus (int): Primes exclude multiplied together.
        candidates (list[int]): Candidate primes within the modulus.
        master_loop (list[list[int]]): List of prime numbers. Need when all arrays aren't avaiable.
        add (list[list[int]]): sorted sublists and transposed, passed individually for performing parallel sieve in specific residue class."""
    @staticmethod
    def gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a

    def __init__(self, n, exclude):
        """ initializes object's attributes.
                .limit .modulus .candidates .master_loop .add

            Args:
                n (int): Limit of the sieve. Used to define master_loop.
                exclude (list[int]): Prime factors excluded from sieving."""
        self.limit = n
        self.modulus = 1
        for element in exclude:
            self.modulus *= element
        self.candidates = [i for i in range(1, self.modulus) if self.gcd(i, self.modulus) == 1]

        # List of primes == true, referring to the index of candidates.
        primes = list(sieve_eratosthene(int(n ** 0.5) + 1, False))  # Local use: Generate a list of prime numbers less than (n ** 0.5) + 1
        self.master_loop = [[] for _ in range((primes[-1] // self.modulus) + 1)]
        [self.master_loop[_ // self.modulus].append(self.candidates.index(_ % self.modulus)) for _ in primes[len(exclude):]]

        # list used for performing calculations in specific bitarray. load or make and save that list
        config_file_path = config_path + f"/{exclude}.cfg"
        if os.path.exists(config_file_path):                                        # Load constant ADD
            self.add = pickle.load(open(config_file_path, "rb"))
        else:                                                                       # create constant ADD if not
            self.add = []
            for i, value in enumerate(self.candidates):
                table_seq = [x + self.modulus for x in self.candidates[:i]] + self.candidates[i:]  # reconstruct candidates adding modulus for lower items than value.
                self.add.append([item[1] for item in sorted(((x * value) % self.modulus, (x - value) >> 1) for x in table_seq)])  # sorted list with 1st tuple, keep only the 2nd tuple.
            self.add = list(zip(*self.add))  # Transpose the sublists to group elements by index.
            os.makedirs(config_path) if not os.path.exists(config_path) else None  # Make dirpath if don't exist
            pickle.dump(self.add, open(config_file_path, "wb"))                    # Saving constant ADD


def sieve_residue_class(init, i):
    """Perform sieve calculations for one residue class set.

    Args:
        init: object holding constant (limit master_loop modulus candidates)
        i: Index used to get {keys} from candidates, representing the specific residue class.

    Returns:
        Tuple (specific residue class as keys, bitarays)"""
    value = bitarray()
    value.extend(bitarray('1') * (init.limit // init.modulus + 1))
    for nth in range(len(init.master_loop)):                    # Iterate each prod_exclude range
        for ix in init.master_loop[nth]:                       # Iterate the sublist to know if prime, datas are the index in candidates.
            step = nth * init.modulus + init.candidates[ix]   # Convert data into real number
            value[(step ** 2 + (step * init.add[i][ix] << 1)) // init.modulus::step] = 0
    return init.candidates[i], value


def generate_splited(psize, candidates, modulus, part):
    lenarray = len(part[1])
    f = [_ * modulus for _ in range(psize, psize + lenarray)]
    lst = [f[_] + c for _ in range(lenarray) for c in candidates if part[c][_]]
    return lst


def sieve_multicore_residue_class_set(limit, exclude, core, parts):
    bench2 = Chrono()
    init = Constants(limit, exclude)        # initialisation
    print(f"\nLimit: {limit}  {limit:.4e} \n\nSieving & generation of {len(init.candidates)} primes candidate in residue class set of \033[92m{init.modulus} (modulus) \033[0m"
          f"splited into {parts} segments joined in 1 list . on {core} core, \n\033[92mMemory use: {((limit // init.modulus) * len(init.candidates)) /1014/1024:.2f} MB\033[0m by Antoine")

    with Pool(core) as pool:                # multicore sieve
        results = dict(pool.starmap(sieve_residue_class, [(init, i) for i in range(len(init.candidates))]))
    results[1][0] = 0
    print(f"sieve:            {bench2.mark()}")

    part_size = len(results[1]) // parts     # split array for multicore gen
    partitions = {i + 1: {key: results[key][i * part_size:(i + 1) * part_size if i < parts - 1 else None] for key in init.candidates} for i in range(parts)}
    print(f"split:            {bench2.mark()}")

    with Pool(processes=core) as pool:      # multicore gen
        args = [(segment * part_size, init.candidates, init.modulus, partitions[segment + 1]) for segment in range(parts)]
        results = pool.starmap(generate_splited, args)
    while int(results[-1][-1]) > limit:     # remove extra numbers of the last residue class set
        results[-1].pop()
    print(f"Generate in list: {bench2.mark()}  {len(exclude) + sum(len(result) for result in results)}\nRunning time:     {bench2.mark(True)}\n")

    return exclude + [item for result in results for item in result]


if __name__ == "__main__":
    config_path = "c:/optimus"      # path of precal table ADD  take some time for big table that's i save
    limite = 100000000000
    exclusion = [2, 3, 5, 7, 11]    # Exclude numbers, need to be primes and consecutive. more you use egal less memory usage for sieving. but shorter the bitarray. need big limits to be efficient. best up to 7 or 11
    ant = sieve_multicore_residue_class_set(limite, exclusion, core=12, parts=100)   # number of core(process) max 61, and how many time all arrays are splited. create a list, so need alot of memory for big limit

    bench = Chrono()
    era = (sieve_eratosthene(limite))  # simple
    for gen_prime, list_prime in zip(era, (_ for _ in ant)):  # firstly the list "ant" is converted into a generator for compare both as generator
        if gen_prime != list_prime:                           # even with the use of generator it's a very long process for big range multicore is realy more quick
            msg = "Generator aren't identical."
            break
    else:
        msg = "Both generator are identical."
    print(f"Generator: {bench.mark()}\nRunning time:     {bench.mark(True)}\n\n{msg}")
