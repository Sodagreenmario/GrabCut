//
// Created by Sodagreenmario on 2020-05-19.
//

#ifndef GRABCUT_ADAPTEDGRAPH_H
#define GRABCUT_ADAPTEDGRAPH_H

#include "graph.h"

class AdaptedGrpah{
private:
    Graph<double, double, double> *graph;
public:
    AdaptedGrpah() {}
    AdaptedGrpah(int vtxCount, int edgeCount);
    int addVtx();
    void addEdges(int i, int j, double w, double revw);
    void addTermWeights(int i, double sourcew, double sinkw);
    void maxFlow();
    bool inSourceSegment(int i);
};

#endif //GRABCUT_ADAPTEDGRAPH_H
