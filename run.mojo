fn main():
	let mut args = env::args();
	let _ = args.next();
	let n: usize = args.next().unwrap().parse().unwrap();
	let mut a: Vec<usize> = args.map(|x| x.parse().unwrap()).collect();
	a.sort();
	let mut ans = 0;
	for i in 0..n {
		ans += a[i * 2];
	}
	println!("{}", ans);
