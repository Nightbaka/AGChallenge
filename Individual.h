#pragma once

#include <vector>
using namespace std;



class Individual
{
public:
	Individual(vector<int> * genome) {
		genes = genome;
	}
	~Individual() {
		delete genes;
	}
	Individual(Individual& other) {
		genes = new vector<int>(*(other.genes));
	}

	float score;
	vector<int>* genes;

	void print() {
		for (size_t i = 0; i < genes->size(); i++)
		{
			cout << genes->at(i);
		}
		cout << endl;
	}
};

