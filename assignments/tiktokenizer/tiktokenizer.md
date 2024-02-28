#code to tokenize
#---------------------------------
def uniform_cost_search(graph, startNode, endKey, heuristics = None):
    
  frontier = PriorityQueue()
  reached = {}
  # I have a bug where the frontier still keeps the old nodes, so I clearit
  frontier.clear()

  reached[startNode.key] = startNode
  frontier.append(startNode)

  while(not (frontier.isEmpty())):
      
    node = frontier.pop()   
    print("\nExpanded Node: ", node, "\n")
    
    # Apply heuristic if available and not at start node
    if heuristics is not None and node.key != startNode.key:
        # heuristic subtracted so new heuristic from successor node can be added
        node.costRoot = node.costRoot - heuristics[node.key] 
    
    # Check if destination node is reached
    if node.key == endKey:
        return node

    # Expand the current node and update reached nodes
    for child in expand(graph, node, heuristics):
        if((not child.key in reached) or (child.costRoot < reached[child.key].costRoot)):
            reached[child.key] = child
            frontier.append(child)
            
    frontier.print()


#tokenized
#---------------------------------

[755, 14113, 16269, 10947, 25301, 11, 1212, 1997, 11, 842, 1622, 11, 568, 324, 5706, 284, 2290, 997, 
1084, 262, 49100, 284, 85202, 746, 262, 8813, 284, 5731, 262, 674, 358, 617, 264, 10077, 1405, 279, 
49100, 2103, 13912, 279, 2362, 7954, 11, 779, 358, 2867, 275, 198, 262, 49100, 7578, 2892, 262, 8813, 
29563, 1997, 4840, 60, 284, 1212, 1997, 198, 262, 49100, 2102, 10865, 1997, 696, 262, 1418, 25804, 320, 
7096, 1291, 9389, 368, 10162, 1827, 286, 2494, 284, 49100, 8452, 368, 5996, 286, 1194, 5026, 77, 53033, 
6146, 25, 3755, 2494, 11, 2990, 77, 1158, 1827, 286, 674, 21194, 67709, 422, 2561, 323, 539, 520, 1212, 
2494, 198, 286, 422, 568, 324, 5706, 374, 539, 2290, 323, 2494, 4840, 976, 1212, 1997, 4840, 512, 310, 
674, 67709, 33356, 291, 779, 502, 67709, 505, 34665, 2494, 649, 387, 3779, 198, 310, 2494, 40266, 8605, 
284, 2494, 40266, 8605, 482, 568, 324, 5706, 30997, 4840, 60, 51087, 286, 674, 4343, 422, 9284, 2494, 374, 
8813, 198, 286, 422, 2494, 4840, 624, 842, 1622, 512, 310, 471, 2494, 271, 286, 674, 51241, 279, 1510, 2494, 
323, 2713, 8813, 7954, 198, 286, 369, 1716, 304, 9407, 25301, 11, 2494, 11, 568, 324, 5706, 997, 310, 422, 
1209, 1962, 1716, 4840, 304, 8813, 8, 477, 320, 3124, 40266, 8605, 366, 8813, 86583, 4840, 948, 16845, 8605, 
10162, 394, 8813, 86583, 4840, 60, 284, 1716, 198, 394, 49100, 2102, 18272, 340, 6494, 286, 49100, 2263, 368]


#In your own words, describe with the tokenized description how LLMs, tokenizers, mixture of experts, and Coding Assistance fits together.
#---------------------------------

Tokenizers break down the input of users into tokens that are fragments of the sentences. Tokens can represent single characters or
whole words. LLMs use these tokenizers to be able to "understand" the text and how to interpret the text. It determines the context of 
the text and how the LLM will ultimately respond to it. A mixture of experts is the method in how tokenizers work, and different LLMs have 
different methods of tokenizing strings depending on how advanced / informed a model is. With the code, I put into the tokenizer, it broke
down each line into relatively large tokens, as code is naturally blocky/less flexible by nature. It is easier for LLM's to help with coding
assistance because tokenizing code is much more simple. There are more strict rules about coding syntax/conventions, and it is very uniform/
consistent. It's very easy to group up different types of syntax/code and relate them to each other since there's far less variation. And the
variation that there is is clearly distinct / defined.
