#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <SFML/Graphics.hpp>
class node{
  public:
      int i;
      int x;
      int y;
};

class graph{
      int n;
      std::vector<std::vector<int>> adjacencyList;
      std::vector<node> nodes;
  public:
         graph(int n){
        this->n = n;
        adjacencyList.resize(n);
        nodes.resize(n);
      }

      void addEdge(int src, int dest){
        adjacencyList[src].push_back(dest);
      }

      void addNode(int nodeIndex, double x, double y) {
        nodes[nodeIndex].i = nodeIndex;
        nodes[nodeIndex].x = x;
        nodes[nodeIndex].y = y;
      }

    void printGraph(){
      for(int i = 0; i < this->n; i++){
        std::cout << "No de numero " << i << " -> X: " << nodes[i].x << " | Y: " << nodes[i].y << " | Nos Proximos ";
        for(auto prox : adjacencyList[i]){
          std::cout << prox << " ";
        }
        std::cout << std::endl;
      }
    }

    double nodeDistance(node A, node B){
      double distancia = sqrt(pow(A.x - B.x, 2) + pow(A.y - B.y, 2));
      return distancia;
    }

    void setSmallWorld(int n, double p){
      for(int i = 0; i < this->n; i++){
        closestNodes(i, nodes, n);
      }

      for(int i = 0; i < this->n; i++){
        setRandomize(i, adjacencyList, p, this->n);
      }

    }

    void setRandomize(int nodeIndex,std::vector<std::vector<int>> &adjacents , double p, int size){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 100);
        for(int i = 0; i < adjacents[nodeIndex].size(); i++){
            double change = dis(gen);
            if(change < p*100){
              std::uniform_int_distribution<> definal(0, size);
              int newVal = 0;
              do{
                newVal = definal(gen);
              }while( (newVal == nodeIndex || newVal == adjacents[nodeIndex][i]) && std::find(adjacents[nodeIndex].begin(), adjacents[nodeIndex].end(), newVal) == adjacents[nodeIndex].end());
            //  std::cout << "No " << nodeIndex  << ":Mudou do valor " << adjacents[nodeIndex][i] << " para o nó " << newVal << std::endl;
              adjacents[nodeIndex][i] = newVal;

             // std::cout<< "Prova: " << adjacents[nodeIndex][i] << std::endl;
            }
        }

    }
    void closestNodes(int nodeIndex, std::vector<node> nodeList, int n){
      node nodeA = nodeList[nodeIndex];
      std::vector<double> distance;
      for(int i = 0; i < nodeList.size(); i++){
        if(i != nodeA.i)
          distance.push_back(nodeDistance(nodeA, nodeList[i]));
        else{
          distance.push_back(INFINITY);
        }
      }
      /*
      std::cout << "Setando distancia do node " << nodeA.i << ": ";
      for (int i = 0; i < distance.size(); i++) {
        std::cout << "para " << i << " :"  << distance[i] << std::endl;

    }

      std::cout << "setado" << std::endl;*/
      std::vector<int> indices(distance.size());
      for (int i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }

      std::partial_sort(indices.begin(), indices.begin() + n, indices.end(),
                 [&](int i, int j) { return distance[i] < distance[j]; });

      for (int i = 0; i < n; i++) {
        //std::cout << indices[i] << " ";
        addEdge(nodeA.i, indices[i]);
    }
   }
    void plot() {
        // Create window and set background color
        sf::RenderWindow window(sf::VideoMode(800, 800), "Graph Plot");
        window.clear(sf::Color::White);

        // Draw axes
        sf::VertexArray axes(sf::Lines, 2);
        axes[0].position = sf::Vector2f(400, 0);
        axes[1].position = sf::Vector2f(400, 800);
        axes[0].color = sf::Color::Black;
        axes[1].color = sf::Color::Black;
        window.draw(axes);
        axes[0].position = sf::Vector2f(0, 400);
        axes[1].position = sf::Vector2f(800, 400);
        window.draw(axes);

        // Draw axis labels
        sf::Font font;
        if (!font.loadFromFile("arial.ttf")) {
            std::cout << "Error loading font" << std::endl;
            return;
        }
        sf::Text label;
        label.setFont(font);
        label.setCharacterSize(14);
        label.setFillColor(sf::Color::Black);
        label.setStyle(sf::Text::Regular);
        for (int i = -10; i <= 10; ++i) {
            if (i == 0) continue; // Don't label origin
            label.setString(std::to_string(i*10));
            label.setPosition(400 + i * 40, 400);
            window.draw(label);
            label.setPosition(400, 400 - i * 40);
            window.draw(label);
        }

        // Draw nodes
        sf::CircleShape nodeShape(5.f);
        nodeShape.setFillColor(sf::Color::Black);
        for (const auto& node : nodes) {
            nodeShape.setPosition(400 + node.x * 4, 400 - node.y * 4);
            window.draw(nodeShape);
        }

        // Draw edgesi
         // Draw edges
sf::Vertex line[2];
line[0].color = sf::Color::Red;
line[1].color = sf::Color::Red;
for (int i = 0; i < nodes.size(); ++i) {
    const auto& adjNodes = adjacencyList[i];
    for (const auto& adjNode : adjNodes) {


        sf::Vertex line[2];
        line[0].color = sf::Color::Red;
        line[1].color = sf::Color::Red;
        for (int i = 0; i < nodes.size(); ++i) {
            const auto& adjNodes = adjacencyList[i];
            for (const auto& adjNode : adjNodes) {
                line[0].position = sf::Vector2f(400 + nodes[i].x * 4, 400 - nodes[i].y * 4);
                line[1].position = sf::Vector2f(400 + nodes[adjNode].x * 4, 400 - nodes[adjNode].y * 4);
                window.draw(line, 2, sf::Lines);
            }

        }

    }
}
        // Display window
        sf::Texture texture;
        texture.create(window.getSize().x, window.getSize().y);
        texture.update(window);
        sf::Image screenshot = texture.copyToImage();
        std::string filename = "screenshot.png";
        screenshot.saveToFile(filename);

    }
    bool dfsUtil(int curr, int destination, std::vector<bool>& visited, std::vector<int>& parent) {
        visited[curr] = true;

        if (curr == destination) {
            return true;
        }

        for (int neighbor : adjacencyList[curr]) {
            if (!visited[neighbor]) {
                parent[neighbor] = curr;
                if (dfsUtil(neighbor, destination, visited, parent)) {
                    return true;
                }
            }
        }

        return false;
    }

    void dfs(int source, int destination) {
      std::vector<bool> visited(n, false);
      std::vector<int> parent(n, -1);

        bool found = dfsUtil(source, destination, visited, parent);

        if (!found) {
          std::cout << "Destination node not reachable from source node" << std::endl;
            return;
        }

        std::vector<int> path;
        int curr = destination;
        while (curr != -1) {
            path.push_back(curr);
            curr = parent[curr];
        }

        std::cout << "Path from node " << source << " to node " << destination << ":" << std::endl;
        for (int i = path.size() - 1; i >= 0; i--) {
          std::cout << path[i] << " ";
        }
        std::cout << std::endl;
    }

    };

//g++ main.cpp -o main -lsfml-graphics -lsfml-window -lsfml-system -limplot


int main(){
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 100);
  int n = 2000;
  graph* g = new graph(n);
  for (int i = 0; i < n; i++){
    int x = dis(gen);
    int y = dis(gen);
    g->addNode(i , x, y);
  }
    g->setSmallWorld(5, 0.1);

  //g->printGraph();
 //g->plot();
 //
 g->dfs(1, 2);
//g->printGraph();
}
