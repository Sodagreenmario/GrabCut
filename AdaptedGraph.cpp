//
// Created by Sodagreenmario on 2020-05-19.
//

#include "AdaptedGraph.h"

AdaptedGrpah::AdaptedGrpah(int vtxCount, int edgeCount){
    graph = new Graph<double, double, double>(vtxCount, edgeCount);
}

int AdaptedGrpah::addVtx(){
    return graph->add_node(1);
}

void AdaptedGrpah::addEdges(int i, int j, double w, double revw){
    graph->add_edge(i, j, w, revw);
}
void AdaptedGrpah::addTermWeights(int i, double sourcew, double sinkw){
    graph->add_tweights(i, sourcew, sinkw);
}
void AdaptedGrpah::maxFlow(){
    graph->maxflow();
}

bool AdaptedGrpah::inSourceSegment(int i){
    return graph->what_segment(i) == Graph<double, double, double>::SOURCE;
}