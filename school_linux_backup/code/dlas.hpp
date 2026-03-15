#ifndef DLAS_HPP

#include "intdef.hpp"
#include <array>
#include <algorithm>
#include <functional>
#include <utility>

template<class T, class U>
T incMod(T x, U mod) {
	x += 1;
	return x == mod ? 0 : x;
}

template<class Domain, class CoDomain, class F, class Mutate, size_t LEN = 5>
std::pair<Domain, CoDomain> dlas(
	F f,
	Mutate mutate,
	Domain const& initial,
	u64 maxIdleIters = -1ULL
) {
	std::array<Domain, 3> S{initial, initial, initial};
	CoDomain curF = f(S[0]);
	size_t curPos = 0;
	size_t minPos = 0;

	std::array<CoDomain, LEN> fitness;
	std::fill(fitness.begin(), fitness.end(), curF);
	CoDomain minF = curF;
	size_t k = 0;

	for (u64 idleIters = 0; idleIters < maxIdleIters; idleIters += 1) {
		CoDomain prvF = curF;

		size_t newPos = incMod(curPos, 3);
		if (newPos == minPos) newPos = incMod(newPos, 3);

		Domain& curS = S[curPos];
		Domain& newS = S[newPos];

		newS = curS;
		mutate(newS);
		CoDomain newF = f(newS);
		if (newF < minF) {
			idleIters = 0;
			minPos = newPos;
			minF = newF;
		}
		if (newF == curF || newF < *std::max_element(fitness.begin(), fitness.end())) {
			curPos = newPos;
			curF = newF;
		}

		CoDomain& fit = fitness[k];
		if (curF > fit || curF < fit && curF < prvF) {
			fit = curF;
		}
		k = incMod(k, LEN);
	}
	return { S[minPos], minF };
}

#define DLAS_HPP
#endif

