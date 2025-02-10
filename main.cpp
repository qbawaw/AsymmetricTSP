#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <limits>
#include <cmath>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>

using json = nlohmann::json;

struct Ant {
    std::vector<int> path;
    double cost;
};

std::vector<std::vector<double>> readGraph(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }
    json graphJson;
    file >> graphJson;
    file.close();

    size_t n = graphJson.size();
    std::vector<std::vector<double>> graph(n, std::vector<double>(n, std::numeric_limits<double>::infinity()));
    
    for (const auto& [from, neighbors] : graphJson.items()) {
        int fromIdx = std::stoi(from);
        for (const auto& [to, weight] : neighbors.items()) {
            int toIdx = std::stoi(to);
            graph[fromIdx][toIdx] = weight;
        }
    }

    return graph;
}

std::vector<std::vector<double>> generateRandomGraph(size_t n, double minWeight = 1.0, double maxWeight = 10.0) {
    std::vector<std::vector<double>> graph(n, std::vector<double>(n, std::numeric_limits<double>::infinity()));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(minWeight, maxWeight);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i != j) {
                graph[i][j] = dis(gen);
            }
        }
    }

    return graph;
}

template<typename Func>
double measureTime(Func func, const std::vector<std::vector<double>>& graph) {
    auto start = std::chrono::high_resolution_clock::now();
    func(graph);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

std::pair<int, int> makeKey(int vertex, int mask) {
    return std::make_pair(vertex, mask);
}

double heldKarpHelper(int currentVertex, int mask, const std::vector<std::vector<double>>& graph, std::map<std::pair<int, int>, std::pair<double, std::vector<int>>>& dp) {
    if (mask == (1 << graph.size()) - 1) {
        return graph[currentVertex][0];
    }

    auto key = makeKey(currentVertex, mask);
    if (dp.find(key) != dp.end()) {
        return dp[key].first;
    }

    double minCost = std::numeric_limits<double>::max();
    std::vector<int> bestPath;

    for (int nextVertex = 0; nextVertex < graph.size(); ++nextVertex) {
        if (!(mask & (1 << nextVertex))) {
            double cost = graph[currentVertex][nextVertex] + heldKarpHelper(nextVertex, mask | (1 << nextVertex), graph, dp);
            if (cost < minCost) {
                minCost = cost;
                bestPath = dp[makeKey(nextVertex, mask | (1 << nextVertex))].second;
                bestPath.push_back(nextVertex);
            }
        }
    }

    dp[key] = std::make_pair(minCost, bestPath);
    return minCost;
}

std::pair<double, std::vector<int>> heldKarp(const std::vector<std::vector<double>>& graph) {
    std::map<std::pair<int, int>, std::pair<double, std::vector<int>>> dp;
    double minCost = heldKarpHelper(0, 1, graph, dp);

    std::vector<int> path = dp[makeKey(0, 1)].second;
    std::reverse(path.begin(), path.end());
    path.insert(path.begin(), 0);
    path.push_back(0);

    return std::make_pair(minCost, path);
}

std::pair<double, std::vector<int>> repetitiveNearestNeighbour(const std::vector<std::vector<double>>& graph) {
    size_t n = graph.size();
    double bestCost = std::numeric_limits<double>::infinity();
    std::vector<int> bestPath;

    for (size_t start = 0; start < n; ++start) {
        std::vector<bool> visited(n, false);
        std::vector<int> path;
        double cost = 0;
        int current = start;
        path.push_back(current);
        visited[current] = true;

        for (size_t i = 1; i < n; ++i) {
            double minDist = std::numeric_limits<double>::infinity();
            int nextNode = -1;
            for (size_t j = 0; j < n; ++j) {
                if (!visited[j] && graph[current][j] < minDist) {
                    minDist = graph[current][j];
                    nextNode = j;
                }
            }
            cost += minDist;
            current = nextNode;
            path.push_back(current);
            visited[current] = true;
        }
        cost += graph[current][start];
        path.push_back(start);

        if (cost < bestCost) {
            bestCost = cost;
            bestPath = path;
        }
    }
    return {bestCost, bestPath};
}

std::pair<double, std::vector<int>> antColonyOptimization(const std::vector<std::vector<double>>& graph, int numAnts = 20, int iterations = 100, double alpha = 1.0, double beta = 2.0, double evaporation = 0.5, double Q = 100.0) {
    int n = graph.size();
    std::vector<std::vector<double>> pheromones(n, std::vector<double>(n, 1.0));
    std::vector<int> bestPath;
    double bestCost = std::numeric_limits<double>::infinity();
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Ant> ants(numAnts);

        for (auto& ant : ants) {
            ant.path.push_back(0);
            std::vector<bool> visited(n, false);
            visited[0] = true;
            double cost = 0.0;

            for (int step = 1; step < n; ++step) {
                int current = ant.path.back();
                std::vector<double> probabilities(n, 0.0);
                double sum = 0.0;

                for (int j = 0; j < n; ++j) {
                    if (!visited[j] && graph[current][j] != std::numeric_limits<double>::infinity()) {
                        probabilities[j] = std::pow(pheromones[current][j], alpha) * std::pow(1.0 / graph[current][j], beta);
                        sum += probabilities[j];
                    }
                }

                if (sum == 0) break;

                std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
                int next = dist(gen);
                ant.path.push_back(next);
                visited[next] = true;
                cost += graph[current][next];
            }

            if (ant.path.size() == n && graph[ant.path.back()][0] != std::numeric_limits<double>::infinity()) {
                ant.path.push_back(0);
                cost += graph[ant.path[n - 1]][0];
            } else {
                cost = std::numeric_limits<double>::infinity();
            }

            ant.cost = cost;

            if (cost < bestCost) {
                bestCost = cost;
                bestPath = ant.path;
            }
        }

        for (auto& row : pheromones) {
            for (auto& p : row) {
                p *= (1.0 - evaporation);
            }
        }

        for (const auto& ant : ants) {
            if (ant.cost < std::numeric_limits<double>::infinity()) {
                for (size_t i = 0; i < ant.path.size() - 1; ++i) {
                    int from = ant.path[i];
                    int to = ant.path[i + 1];
                    pheromones[from][to] += Q / ant.cost;
                }
            }
        }
    }

    return {bestCost, bestPath};
}

void saveResults(const std::string& filename, const std::vector<std::pair<std::string, std::pair<double, std::vector<int>>>>& results) {
    json resultJson;
    for (const auto& [method, result] : results) {
        resultJson[method]["cost"] = result.first;
        resultJson[method]["path"] = result.second;
    }
    std::ofstream file(filename);
    file << resultJson.dump(4);
}

void testHeldKarp(const std::string& outputFilename) {
    std::ofstream outputFile(outputFilename);
    if (!outputFile.is_open()) {
        std::cerr << "Nie można otworzyć pliku do zapisu: " << outputFilename << std::endl;
        return;
    }

    outputFile << "Size,Held-Karp Time\n";

    for (size_t size = 1; size <= 16; ++size) {
        std::cout << "Testowanie Held-Karp dla rozmiaru: " << size << std::endl;

        auto graph = generateRandomGraph(size);

        double hkTime = measureTime([](const std::vector<std::vector<double>>& g) { heldKarp(g); }, graph);

        outputFile << size << "," << std::fixed << std::setprecision(6) << hkTime << "\n";
    }

    outputFile.close();
    std::cout << "Testy Held-Karp zakończone. Wyniki zapisane w pliku: " << outputFilename << std::endl;
}

void testRepetitiveNearestNeighbour(const std::string& outputFilename) {
    std::ofstream outputFile(outputFilename);
    if (!outputFile.is_open()) {
        std::cerr << "Nie można otworzyć pliku do zapisu: " << outputFilename << std::endl;
        return;
    }

    outputFile << "Size,RNN Time\n";

    for (size_t size = 2; size <= 400; size += 2) {
        std::cout << "Testowanie Repetitive Nearest Neighbour dla rozmiaru: " << size << std::endl;

        auto graph = generateRandomGraph(size);

        double rnnTime = measureTime([](const std::vector<std::vector<double>>& g) { repetitiveNearestNeighbour(g); }, graph);

        outputFile << size << "," << std::fixed << std::setprecision(6) << rnnTime << "\n";
    }

    outputFile.close();
    std::cout << "Testy Repetitive Nearest Neighbour zakończone. Wyniki zapisane w pliku: " << outputFilename << std::endl;
}

void testAntColonyOptimization(const std::string& outputFilename) {
    std::ofstream outputFile(outputFilename);
    if (!outputFile.is_open()) {
        std::cerr << "Nie można otworzyć pliku do zapisu: " << outputFilename << std::endl;
        return;
    }

    outputFile << "Size,ACO Time\n";

    for (size_t size = 1; size <= 100; size += 1) {
        std::cout << "Testowanie Ant Colony Optimization dla rozmiaru: " << size << std::endl;

        auto graph = generateRandomGraph(size);

        double acoTime = measureTime([](const std::vector<std::vector<double>>& g) { antColonyOptimization(g); }, graph);

        outputFile << size << "," << std::fixed << std::setprecision(6) << acoTime << "\n";
    }

    outputFile.close();
    std::cout << "Testy Ant Colony Optimization zakończone. Wyniki zapisane w pliku: " << outputFilename << std::endl;
}

int main() {
    auto graph = readGraph("graph.json");
    saveResults("results.json", {
        {"Held-Karp", heldKarp(graph)},
        {"Repetitive Nearest Neighbour", repetitiveNearestNeighbour(graph)},
        {"Ant Colony Optimization", antColonyOptimization(graph)}
    });

    //testHeldKarp("held_karp_times.csv");
    //testRepetitiveNearestNeighbour("rnn_times.csv");
    //testAntColonyOptimization("aco_times.csv");
}
