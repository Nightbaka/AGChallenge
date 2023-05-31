#pragma once

#include "Evaluator.h"

#include <random>
#include <vector>


using namespace std;

class COptimizer
{
public:
	COptimizer(CEvaluator &cEvaluator);
	~COptimizer();

	void vInitialize();
	void vRunIteration();

	vector<int> *pvGetCurrentBest() { return &v_current_best; }

private:
	void v_fill_randomly(vector<int> &vSolution);

	CEvaluator &c_evaluator;

	double d_current_best_fitness;
	vector<int> v_current_best;

	mt19937 c_rand_engine;

	//my additions
	void updateBest(vector<int> & newBest, double fitness);
	double climbHill(vector<int>& individual);
	bool inject(vector<int>& cluster, vector<int>& solution, vector<int> donor, double& fitness);
	void buildClusters(int level);
	int genomeLength;
	int currentIteration;
	struct VectorHasher {
		int operator()(const vector<int>& v) const {
			int hash = v.size();
			for (auto& i : v) {
				hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
			}
			return hash;
		}
	};
	double getDistance(array<int, 4>& bitMemory);
	double getDistance(int x, int y, int level);
	bool addSolution(vector<int> & solution, int level, double fitness);
	bool crossWithLevel(int level, vector<int> & solution, double & fitness);
	void updateAdjacencyMatrix(vector<int>& solution, int level);
	void p3Iteration();
	void addLevel();
	//entropy
	double entropy(array<int, 4>& counts, double total);

	unordered_map<vector<int>, double, VectorHasher> solutionsMap;
	vector<vector<vector<array<int, 4>>>*> adjacencyMatrix;
	vector<vector<vector<int>*>*> pyramid;
	vector<vector<vector<int>*>*> pyramidOfClusters;
	vector<vector<int>*> pyramidOfUsefulClusters;
	vector<vector<double>> distances;
	int maxPopulation;
	int minPopulationForCrossover;

	//determining problem utility
	int sampleSize = 100;
	double averageRandomSolution();
	double currentRandomSum;
	double averageOptimizedSolution();
	double currentOptimizedSum;
	double averagaRandomVariance();
	vector<double> randomValueStorage;
	double averageOptimizedVariance();
	vector<double> optimizedValueStorage;
	double getVariance(vector<double>& values);
};//class COptimizer