#include <bits/stdc++.h>

using namespace std;


struct Connection{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;


//************************** class Neuron ***********************
class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned ind);
    void setOutput(double value) { output = value; }
    double getOutput(void) const { return output; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetValue);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
private:
    static double eta; // something from 0 - 1
    static double alpha; // 0 - n, multiplier of last weight change
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void){ return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double output;
    vector<Connection> outputWeights;
    unsigned index;
    double gradient;
};

double Neuron::eta = 0.15; // learning rate
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight



void Neuron::updateInputWeights(Layer &prevLayer){
    // the weights to be updated are in the connection container in the neurons in the preceding layer
    for(unsigned i = 0; i < prevLayer.size(); ++i){
        Neuron &neuron = prevLayer[i];
        double oldDeltaWeight = neuron.outputWeights[index].weight;
        double newDeltaWeight =
            // individual input, magnified by the gradient and train rate
            eta
            * neuron.getOutput()
            * gradient
            // also add momentum = a fraction of the previous delta weight
            + alpha
            * oldDeltaWeight;

        neuron.outputWeights[index].deltaWeight = newDeltaWeight;
        neuron.outputWeights[index].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const{
    double sum = 0.0;

    // sum our contributions of the errors at the nodes we feed

    for(unsigned i = 0; i < nextLayer.size(); ++i){
        sum += outputWeights[i].weight * nextLayer[i].gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    gradient = dow * Neuron::transferFunctionDerivative(output);
}

void Neuron::calcOutputGradients(double targetValue){
    double delta = targetValue - output;
    gradient = delta * Neuron::transferFunctionDerivative(output);
}

double Neuron::transferFunction(double x){
    // tanh - output range [-1.0 - 1.0]
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x){
    // not accurate but fast
    return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer){
    double sum = 0.0;

    for(unsigned i = 0; i < prevLayer.size(); ++i){
        sum += prevLayer[i].getOutput() *
        prevLayer[i].outputWeights[index].weight;
    }
    sum /= prevLayer.size();
    output = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned ind){
    index = ind;
    for(unsigned i = 0; i < numOutputs; ++i){
        outputWeights.push_back(Connection());
        outputWeights.back().weight = randomWeight();
    }
}


//************************** class Net *************************
class Net{
public:
    Net(const vector<unsigned> &topology);

    void feedForward(const vector<double> &inputValues);
    void backProp(const vector<double> &targetValues);
    void getResults(vector<double> &resultValues);
private:
    vector<Layer> layers; //layers[layerNum][neuronNum]
    double error;
    double recentAverageError;
    double recentAverageSmoothingFactor = 0; //change later
};

void Net::getResults(vector<double> &resultValues){
    resultValues.clear();
    for(unsigned i = 0; i < layers.back().size() - 1; ++i){
        resultValues.push_back(layers.back()[i].getOutput());
    }
}

void Net::backProp(const vector<double> &targetValues){
    // calculate overall net error (RMS of output neuron errors)
    Layer &outputLayer = layers.back();
    error = 0.0;
    for(unsigned i = 0; i < outputLayer.size() - 1; ++i){
        double delta = targetValues[i] - outputLayer[i].getOutput();
        error += delta*delta;
    }
    error /= (outputLayer.size() - 1);
    error = sqrt(error); // bc RMS

//    cerr << "RMS of errors: " << error << '\n';

    // implement a recent average measurement
    recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

//     cerr << "recent average error: " << recentAverageError << '\n';

    // calculate output layer gradients
    for(unsigned i = 0; i < outputLayer.size(); ++i){
        outputLayer[i].calcOutputGradients(targetValues[i]);
    }


    // calculate gradients on hidden layers
    for(unsigned i = layers.size() - 2; i > 0; --i){
        Layer &hiddenLayer = layers[i];
        Layer &nextLayer = layers[i + 1];

        for(unsigned j = 0; j < hiddenLayer.size(); ++j){
            hiddenLayer[i].calcHiddenGradients(nextLayer);
        }
    }

//    cerr << "Gradients calculated\n";

    // for all layers from outputs to first hidden layer, update connection weights
    for(unsigned i = layers.size() - 1; i > 0; --i){
        Layer &prevLayer = layers[i - 1];
        for(unsigned j = 0; j < layers[i].size() - 1; ++j){
            layers[i][j].updateInputWeights(prevLayer);
//            cerr << "updated input weights for neuron " << i << ',' << j << '\n';
        }
    }
//    cerr << "weights updated\n";


};

void Net::feedForward(const vector<double> &inputValues){
    for(unsigned i = 0; i < inputValues.size(); ++i){
        layers[0][i].setOutput(inputValues[i]);
    }
//    cerr << "set output for all bias neurons\n";

    // forward propagation
    for(unsigned i = 1; i < layers.size(); ++i){
        Layer &prevLayer = layers[i-1];
        for(unsigned j = 0; j < layers[i].size(); ++j){
            layers[i][j].feedForward(prevLayer);
//            cerr << "neuron " << i << ',' << j << " fed forward\n";
        }
    }


};

Net::Net(const vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for(unsigned i = 0; i < numLayers; ++i){
        layers.push_back(Layer());
        unsigned numOutputs = i == topology.size() - 1 ? 0 : topology[i + 1];

        //<= because of the bias Neuron
        for(unsigned j = 0; j <= topology[i]; ++j){
            layers.back().push_back(Neuron(numOutputs, j));
            cerr << "Neuron " << i << ',' << j << " created\n";
        }
        // force the bias neurons output value to 1.0
        layers.back().back().setOutput(1.0);
    }
}


vector<double> inputValues;
vector<double> targetValues;
vector<double> resultValues;
vector<unsigned> topology = {2, 4, 1};
Net net(topology);

void netStuff(void){
    net.feedForward(inputValues);
//    cerr << "fed everything forward\n";
    net.backProp(targetValues);
//    cerr << "propagated back\n";
    net.getResults(resultValues);
    for(unsigned i = 0; i < targetValues.size(); ++i){
        cerr << "target value: " << targetValues[i] << ", result value: " << resultValues[i] << endl;
    }
}




int main(){

    //read input file from other program to test
    ifstream input("input.txt");
    double tmp;

    for(unsigned i = 0; i < 100; ++i){
        inputValues.clear();
        targetValues.clear();
        input >> tmp;
        inputValues.push_back(tmp);
        input >> tmp;
        inputValues.push_back(tmp);
        input >> tmp;
        targetValues.push_back(tmp);
        netStuff();
    }


    return 0;
}
