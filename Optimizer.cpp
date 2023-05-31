#include "Optimizer.h"

#include <cfloat>
#include <iostream>
#include <windows.h>

using namespace std;

COptimizer::COptimizer(CEvaluator &cEvaluator)
	: c_evaluator(cEvaluator)
{
	random_device c_seed_generator;
	c_rand_engine.seed(c_seed_generator());
	d_current_best_fitness = 0;

	//my changes 
}//COptimizer::COptimizer(CEvaluator &cEvaluator)

COptimizer::~COptimizer()
{
	//error
	//delete pyramid of Clusters
	for (size_t i = 0; i < pyramidOfClusters.size(); i++)
	{
		vector<vector<int>*>* cluster;
		cluster = pyramidOfClusters[i];
		for (size_t j = 0; j < cluster->size(); j++)
		{
			delete (*cluster)[j];
		}
		delete cluster;
	}
	//delete pyramid of usefulClusters
	for (size_t i = 0; i < pyramidOfUsefulClusters.size(); i++)
	{
		delete pyramidOfUsefulClusters[i];
	}
	//delete solution pyramid
	for (size_t i = 0; i < pyramid.size(); i++)
	{
		vector<vector<int>*>* solutions;
		solutions = pyramid[i];
		for (size_t j = 0; j < solutions->size(); j++)
		{
			delete (*solutions)[j];
		}
		delete solutions;
	}
	//delete pyramid of adjacencyMatrixes
	for (size_t i = 0; i < adjacencyMatrix.size(); i++)
	{
		vector<vector<array<int, 4>>>* matrix;
		matrix = adjacencyMatrix[i];
		delete matrix;
	}
	cout << c_evaluator.dEvaluate(v_current_best) << endl;
	//deleted everything?
}

void COptimizer::vInitialize()
{
	d_current_best_fitness = -DBL_MAX;
	v_current_best.clear();
	genomeLength = c_evaluator.iGetNumberOfBits();
	currentIteration = 0;
	//first level init
	addLevel();
	maxPopulation = 1000;
	minPopulationForCrossover = 4;
	int clusterSize = (genomeLength * 2) - 1;
	distances.resize(clusterSize, vector<double>(clusterSize, -1));
	int baseSize = 20000;
	solutionsMap.reserve(baseSize);
}//void COptimizer::vInitialize()

void COptimizer::vRunIteration()
{
	p3Iteration();
	currentIteration++;
}//void COptimizer::vRunIteration()

double COptimizer::averageRandomSolution() {
	vector<int> randomVector(genomeLength);
	v_fill_randomly(randomVector);
	currentRandomSum += c_evaluator.dEvaluate(randomVector);
	if (currentIteration == sampleSize) {
		cout << currentRandomSum / (double)100 << endl;
	}
	return currentRandomSum;
}

double COptimizer::averageOptimizedSolution() {
	vector<int> randomVector(genomeLength);
	v_fill_randomly(randomVector);
	climbHill(randomVector);
	currentOptimizedSum += c_evaluator.dEvaluate(randomVector);
	if (currentIteration == sampleSize) {
		cout << currentOptimizedSum / (double)100 << endl;
	}
	return currentOptimizedSum;
}

double COptimizer::averagaRandomVariance() {
	if (currentIteration > sampleSize) return 0.0;
	vector<int> randomVector(genomeLength);
	v_fill_randomly(randomVector);
	randomValueStorage[currentIteration-1] = c_evaluator.dEvaluate(randomVector);
	if (currentIteration == sampleSize)
	{
		cout << getVariance(randomValueStorage) << endl;
	}
}

double COptimizer::averageOptimizedVariance() {
	if (currentIteration > sampleSize) return 0.0;
	vector<int> randomVector(genomeLength);
	v_fill_randomly(randomVector);
	climbHill(randomVector);
	optimizedValueStorage[currentIteration-1] = c_evaluator.dEvaluate(randomVector);
	if (currentIteration == sampleSize)
	{
		cout << getVariance(optimizedValueStorage) << endl;
	}
}

double COptimizer::getVariance(vector<double>& values) {
	double sum = 0;
	for (size_t i = 0; i < values.size(); i++)
	{
		sum += values[i];
	}
	double mean = sum / (double)values.size();
	double varianceSum = 0;
	for (size_t i = 0; i < values.size(); i++)
	{
		varianceSum += (values[i] - mean) * (values[i] - mean);
	}
	double variance = varianceSum / values.size();
	return variance;
}

void COptimizer::p3Iteration() {
	vector<int>* solution;
	solution = new vector<int>();
	v_fill_randomly(*solution);
	double fitness = climbHill(*solution);
	addSolution(*solution, 0, fitness);
	for (size_t i = 0; i < pyramid.size()-1; i++)
	{
		bool improvement = crossWithLevel(i,*solution, fitness);
		if (improvement) {
			addSolution(*solution,i+1,fitness);
			updateBest(*solution, fitness);
		}
	}
	bool improvement = crossWithLevel(pyramid.size()-1, *solution, fitness);
	if (improvement) {
		addLevel();
		addSolution(*solution, pyramid.size()-1, fitness);
	}
	updateBest(*solution, fitness);
	delete solution;
	/*ideas to add to project :
		- limit pop by max Size, move pyramid up when limit reached
		- Apply extra hillclimber after crossover
		- limit nr of times i rebuild clusters
		- change minimum nr for crossover
		- send it before 14!
		- use higher lvl crossover matrix
		- check if the best solution is correctly updated
		- delete all couts
	*/
}

void COptimizer::v_fill_randomly(vector<int> &vSolution)
{
	uniform_int_distribution<int> c_uniform_int_distribution(iBIT_FALSE, iBIT_TRUE);
	vSolution.resize((size_t)c_evaluator.iGetNumberOfBits());

	for (size_t i = 0; i < vSolution.size(); i++)
	{
		vSolution.at(i) = c_uniform_int_distribution(c_rand_engine);
	}
	return;
	//for (size_t i = 0; i < vSolution.size(); i++)
}

double COptimizer::climbHill(vector<int> & genome) {
	vector<int> indexes(genomeLength);
	iota(indexes.begin(), indexes.end(), 0);
	shuffle(indexes.begin(), indexes.end(), c_rand_engine);
	bool improvement = true;
	double fitness = c_evaluator.dEvaluate(genome);
	double newFitness;
	int index;
	while (improvement)
	{
		shuffle(indexes.begin(), indexes.end(), c_rand_engine);
		improvement = false;
		for (size_t i = 0; i < genomeLength; i++)
		{
			index = indexes[i];
			genome[index] = !genome[index];
			newFitness = c_evaluator.dEvaluate(genome);
			if (newFitness > fitness) {
				fitness = newFitness;
				improvement = true;
			}
			else
			{
				genome[index] = !genome[index];
			}
		}
	}
	return fitness;
}

bool COptimizer::inject(vector<int>& cluster, vector<int>& solution, vector<int> donor, double& fitness)
{
	bool changed = false;
	vector<int> tmpStorage(cluster.size());

	for (int i = 0; i < cluster.size(); i++) {
		int index = cluster[i];
		//error
		changed |= (solution[index] != donor[index]);
		tmpStorage[i] = solution[index];
		solution[index] = donor[index];
	}
	if (!changed) { return false; }

	double newFitness = c_evaluator.dEvaluate(solution);
	if (newFitness < fitness)
	{
		//revert changes
		for (size_t i = 0; i < cluster.size(); i++)
		{
			solution[cluster[i]] = tmpStorage[i];
		}
		return false;
	}
	fitness = newFitness;
	return true;
}

void COptimizer::buildClusters(int level)
{
	vector<vector<int>*>& clusters = *pyramidOfClusters[level];
	vector<int> usable(genomeLength);
	iota(usable.begin(), usable.end(), 0);
	shuffle(usable.begin(), usable.end(), c_rand_engine);
	int chainLength = 1;
	boolean wtf = false;
	//calculate distances
	for (int i = 0; i < genomeLength; i++)
	{
		for (int j = i + 1; j < genomeLength; j++) {
			distances[i][j] = getDistance(i,j,level);
			distances[j][i] = distances[i][j];
		}
	}
	//index of merged
	int first, second;
	// used for nearest neighbour search
	int lastInChain, closestNeighbour, lowestDistance;
	vector<bool> useful(clusters.size(), false);
	for (int insertIndex = genomeLength; insertIndex < clusters.size(); insertIndex++)
	{
		if (chainLength < 1) {
			chainLength++;
		}
		while (chainLength < usable.size())
		{
			lastInChain = usable[chainLength - 1];
			int closestIndexInUsable = chainLength;
			closestNeighbour = usable[chainLength];
			lowestDistance = distances[lastInChain][closestNeighbour];
			for (size_t i = chainLength+1; i < usable.size(); i++)
			{
				int candidate = usable[i];
				if (lowestDistance>distances[lastInChain][candidate])
				{
					lowestDistance = distances[lastInChain][candidate];
					closestNeighbour = candidate;
					closestIndexInUsable = i;
				}
			}
			if (chainLength > 1 && distances[usable[chainLength-1]][usable[chainLength-2]] <= lowestDistance)
			{
				break;
			}
			//black magic does work?
			swap(usable[chainLength], usable[closestIndexInUsable]);
			chainLength++;
		}
		//link clusters
		first = usable[chainLength - 1];
		second = usable[chainLength - 2];
		int size1 = clusters[first]->size();
		int size2 = clusters[second]->size();
		vector<int>* previous;
		previous = clusters[insertIndex];
		vector<int>* newCluster;
		newCluster = new vector<int>();
		newCluster->reserve(size1 + size2);
		newCluster->insert(newCluster->end(), clusters[first]->begin(), clusters[first]->end());
		newCluster->insert(newCluster->end(), clusters[second]->begin(), clusters[second]->end());
		clusters[insertIndex] = newCluster;
		delete previous;
		//update distances with new cluster
		for (size_t i = 0; i < insertIndex; i++)
		{
			double newDistance = (size1 * distances[i][first] + size2 * distances[i][second]) / (double)(size1 + size2);
			distances[i][insertIndex] = newDistance;
			distances[insertIndex][i] = newDistance;
		}
		//clean chain - usable
		std::swap(usable[chainLength - 1], usable[usable.size() - 1]);
		std::swap(usable[chainLength - 2], usable[usable.size() - 2]);
		usable.pop_back();
		usable.back() = insertIndex;
		chainLength -= 2;
		//update useful
		useful[insertIndex] = true;
		if (distances[first][second] == 0) {
			useful[first] = false;
			useful[second] = false;
		}
	}
	useful.back() = false;
	vector<int>& usefulClusters = *pyramidOfUsefulClusters[level];
	usefulClusters.clear();
	for (size_t i = genomeLength; i < clusters.size()-1; i++)
	{
		if (useful[i]) {
			usefulClusters.push_back(i);
		}
	}
}

double COptimizer::getDistance(array<int, 4>& bitMemory)
{
	array<int, 4> bitCounts;
	// x = 0
	bitCounts[0] = bitMemory[0] + bitMemory[2];
	// x = 1
	bitCounts[1] = bitMemory[1] + bitMemory[3];
	// y = 0
	bitCounts[2] = bitMemory[0] + bitMemory[1];
	// y = 1
	bitCounts[3] = bitMemory[2] + bitMemory[3];
	int sum = bitMemory[0]+bitMemory[1]+bitMemory[2]+bitMemory[3];

	double sumOfEntropy = entropy(bitCounts, sum);
	double entropyOfSum = entropy(bitMemory, sum);

	double distance = 0;
	if (entropyOfSum == 0) {
		return distance;
	}
	distance = 2 - sumOfEntropy / entropyOfSum;
	return distance;
}

double COptimizer::getDistance(int x, int y, int level)
{
	if (y > x) {
		swap(x, y);
	}
	return getDistance((*(adjacencyMatrix[level]))[y][x]);
}

bool COptimizer::addSolution(vector<int>& solution, int level, double fitness) 
{
	if (solutionsMap.find(solution) != solutionsMap.end())
	{
		return false;
	}
	solutionsMap[solution] = fitness;
	if (pyramid[level]->size() >= maxPopulation) return false;
	updateAdjacencyMatrix(solution, level);
	pyramid[level]->push_back(new vector<int>(solution));
	if (pyramid[level]->size() >= minPopulationForCrossover)
	{
		buildClusters(level);
	}
	return true;
}

bool COptimizer::crossWithLevel(int level, vector<int>& solution, double & fitness) {
	vector<vector<int>*>* population;
	population = pyramid[level];
	int clusterLevel = level;
	while (pyramid[level]->size()<minPopulationForCrossover && clusterLevel > 0)
	{
		clusterLevel--;
	}
	if (pyramid[clusterLevel]->size() < minPopulationForCrossover) return false;
	vector<vector<int>*>& clusters = *pyramidOfClusters[clusterLevel];
	vector<int>& usefulClusters = *(pyramidOfUsefulClusters[clusterLevel]);
	vector<int> popIndexes(population->size());
	iota(popIndexes.begin(), popIndexes.end(), 0);
	int end = popIndexes.size();
	double previousFitness = fitness;
	for (size_t i = 0; i < usefulClusters.size(); i++)
	{
		vector<int>* cluster;
		cluster = clusters[usefulClusters[i]];

		bool improvement = false;
		end = popIndexes.size();
		//get random from pop or go through whole population?
		for (size_t i = 0; !improvement && i < population->size(); i++)
		{
			int lookupIndex = uniform_int_distribution<int>(0, end-1)(c_rand_engine);
			int donorIndex = popIndexes[lookupIndex];
			swap(popIndexes[lookupIndex], popIndexes[end - 1]);
			end--;
			vector<int>* donor;
			donor = (*population)[donorIndex];
			improvement = inject(*cluster, solution, *donor, fitness);
		}
	}
	return previousFitness != fitness;
}

void COptimizer::updateAdjacencyMatrix(vector<int>& solution, int level)
{
	vector<vector<array<int, 4>>>* levelAdjacencyMatrix;
	levelAdjacencyMatrix = adjacencyMatrix[level];
	for (size_t i = 0; i < genomeLength; i++)
	{
		for (size_t j = i + 1; j < genomeLength; j++)
		{
			(*levelAdjacencyMatrix)[i][j][(solution[i] << 1) + solution[j]]++;
		}
	}
}

void COptimizer::addLevel() {
	//adding pop
	vector<vector<int>*>* newPop;
	newPop = new vector<vector<int>*>();
	newPop->reserve(maxPopulation);
	pyramid.push_back(newPop);
	//adding adjacency matrix
	vector<vector<array<int, 4>>>* newAdjacencyMatrix;
	newAdjacencyMatrix = new vector<vector<array<int, 4>>>(genomeLength, vector<array<int, 4>>(genomeLength, array<int, 4>{0,0,0,0}));
	adjacencyMatrix.push_back(newAdjacencyMatrix);
	//adding clusters
	vector<vector<int>*>* newClusters;
	newClusters = new vector<vector<int>*>((genomeLength * 2) - 1);
	for (int i = 0; i < genomeLength; i++)
	{
		(*newClusters)[i] = new vector<int>();
		(*newClusters)[i]->push_back(i);
	}
	for (int i = genomeLength; i < newClusters->size(); i++)
	{
		(*newClusters)[i] = new vector<int>();
	}
	pyramidOfClusters.push_back(newClusters);
	//adding useful clusters
	vector<int>* newUsefulClusters;
	newUsefulClusters = new vector<int>();
	newUsefulClusters->reserve(genomeLength - 1);
	pyramidOfUsefulClusters.push_back(newUsefulClusters);
}

double COptimizer::entropy(array<int, 4>& counts, double total) {
	double entropy = 0;
	double p;
	for (size_t i = 0; i < counts.size(); i++)
	{
		if (counts[i]!=0)
		{
			p = counts[i] / total;
			entropy -= (p * log(p));
		}
	}
	return entropy;
}

void COptimizer::updateBest(vector<int>& newBest, double fitness) {
	if (fitness < d_current_best_fitness)
	{
		return;
	}
	d_current_best_fitness = fitness;
	v_current_best = newBest;
}



//void COptimizer::v_fill_randomly(const vector<int> &vSolution)
