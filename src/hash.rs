use std::hash::Hasher;

pub trait Hash<H: Hasher> {
    fn hash(&self, state: &mut H);
}

pub trait Algorithm<T: AsBytes+Sized> : Hasher {
    fn hash(&self) -> T;

    fn reset(&mut self);
}

pub trait AsBytes {
    fn as_bytes(&self) -> &[u8];
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct MerkleTree<T: AsBytes+Sized, A: Algorithm<T>> {
    data: Vec<T>,
    alg: A,
}

impl<T: AsBytes+Sized, A: Algorithm<T>+Hasher> MerkleTree<T, A> {
    fn new<U: Hash<A>>(data: &[U], alg: A) -> MerkleTree<T, A> {
        let mut t: MerkleTree<T, A> = MerkleTree {
            data: Vec::with_capacity(data.len()),
            alg
        };

        for i in 0..data.len() {
            data[i].hash(&mut t.alg);
            t.data.push(t.alg.hash());
            t.alg.reset();
        }

        t
    }
}

#[cfg(test)]
mod hash_test {
    use super::*;
    use std::fmt;
    use std::hash::Hasher;

    const SIZE:usize = 0x10;

    #[derive(Debug, Copy, Clone)]
    struct Xor128 {
        data: [u8; SIZE],
        i: usize
    }

    impl Xor128 {
        fn new() -> Xor128 {
            Xor128 {
                data: [0; SIZE],
                i: 0
            }
        }
    }

    impl Hasher for Xor128 {
        fn write(&mut self, bytes: &[u8]) {
            for x in bytes {
                self.data[self.i&(SIZE-1)] ^= *x;
                self.i +=1;
            }
        }

        fn finish(&self) -> u64 {
            let mut h : u64 = 0;
            let mut off : u64 = 0;
            for i in 0..8 {
                h |= (self.data[i] as u64) << off;
                off += 8;
            }
            h
        }
    }

    impl AsBytes for [u8; 16] {
        fn as_bytes(&self) -> &[u8] {
            &self[..]
        }
    }

    impl Hash<Xor128> for str {
        fn hash(&self, state: &mut Xor128) {
            state.write(self.as_bytes());
        }
    }

    impl Hash<Xor128> for String {
        fn hash(&self, state: &mut Xor128) {
            state.write(self.as_bytes());
        }
    }

    impl Algorithm<[u8; 16]> for Xor128 {
        fn hash(&self) -> [u8; 16] { self.data }

        fn reset(&mut self) { *self = Xor128::new(); }
    }

    impl fmt::UpperHex for Xor128 {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            if f.alternate() {
                if let Err(e) = f.write_str("0x") {
                    return Err(e)
                }
            }
            for b in self.data.as_bytes() {
                if let Err(e) = write!(f, "{:02X}", b) {
                    return Err(e)
                }
            }
            Ok(())
        }
    }
    
    #[test]
    fn test_hasher_simple() {
        let mut h = Xor128::new();
        "1234567812345678".hash(&mut h);
        h.reset();
        String::from("1234567812345678").hash(&mut h);
        assert_eq!(format!("{:#X}", h), "0xCE323334353637383132333435363738");
        String::from("1234567812345678").hash(&mut h);
        assert_eq!(format!("{:#X}", h), "0xF6FC01070103010F090301070103010F");
        String::from("1234567812345678").hash(&mut h);
        assert_eq!(format!("{:#X}", h), "0xC1C4CF35323734393E3B303532373439");
    }

    #[test]
    fn test_st() {
        let x = [String::from("ars"), String::from("zxc")];
        let mt = MerkleTree::new(&x, Xor128::new());
        assert_eq!(format!("{:?}", mt), "MerkleTree { data: [[97, 114, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [122, 120, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], alg: Xor128 { data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], i: 0 } }");
        assert_eq!(mt.data.len(), 2);
    }
}
