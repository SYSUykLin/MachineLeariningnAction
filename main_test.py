from DecisionTree import DecisionTree

tree = DecisionTree.DecisionTree()
tree.readData("dataSet/datingTestSet2.txt")
node = tree.build_tree()
print(tree.predictTraining())
