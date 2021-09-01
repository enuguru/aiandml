import ID3, parse, random
import matplotlib.pyplot as plt

def testID3AndEvaluate():
  data = [dict(a=1, b=0, Class=1), dict(a=1, b=1, Class=1)]
  tree = ID3.ID3(data, 0)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=1, b=0))
    if ans != 1:
      print "ID3 test failed."
    else:
      print "ID3 test succeeded."
  else:
    print "ID3 test failed -- no tree returned"

def testPruning():
  data = [dict(a=1, b=1, c=1, Class=0), dict(a=1, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1), dict(a=0, b=0, c=0, Class=1), dict(a=0, b=0, c=1, Class=0)]
  validationData = [dict(a=0, b=0, c=1, Class=1)]
  tree = ID3.ID3(data, 0)
  ID3.prune(tree, validationData)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=0, b=1, c=1))
    if ans != 1:
      print "pruning test failed."
    else:
      print "pruning test succeeded."
  else:
    print "pruning test failed -- no tree returned."


def testID3AndTest():
  trainData = [dict(a=1, b=0, c=0, Class=1), dict(a=1, b=1, c=0, Class=1), 
  dict(a=0, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1)]
  testData = [dict(a=1, b=0, c=1, Class=1), dict(a=1, b=1, c=1, Class=1), 
  dict(a=0, b=0, c=1, Class=0), dict(a=0, b=1, c=1, Class=0)]
  tree = ID3.ID3(trainData, 0)
  fails = 0
  if tree != None:
    acc = ID3.test(tree, trainData)
    if acc == 1.0:
      print "testing on train data succeeded."
    else:
      print "testing on train data failed."
      fails = fails + 1
    acc = ID3.test(tree, testData)
    if acc == 0.75:
      print "testing on test data succeeded."
    else:
      print "testing on test data failed."
      fails = fails + 1
    if fails > 0:
      print "Failures: ", fails
    else:
      print "testID3AndTest succeeded."
  else:
    print "testID3andTest failed -- no tree returned."

# inFile - string location of the house data file
def testPruningOnHouseData(inFile):
  withPruning = []
  withoutPruning = []
  data = parse.parse(inFile)
  for i in range(100):
    random.shuffle(data)
    train = data[:len(data)/2]
    valid = data[len(data)/2:3*len(data)/4]
    test = data[3*len(data)/4:]
  
    tree = ID3.ID3(train, 'democrat')
    acc = ID3.test(tree, train)
    print "training accuracy: ",acc
    acc = ID3.test(tree, valid)
    print "validation accuracy: ",acc
    acc = ID3.test(tree, test)
    print "test accuracy: ",acc
  
    ID3.prune(tree, valid)
    acc = ID3.test(tree, train)
    print "pruned tree train accuracy: ",acc
    acc = ID3.test(tree, valid)
    print "pruned tree validation accuracy: ",acc
    acc = ID3.test(tree, test)
    print "pruned tree test accuracy: ",acc
    withPruning.append(acc)
    tree = ID3.ID3(train+valid, 'democrat')
    acc = ID3.test(tree, test)
    print "no pruning test accuracy: ",acc
    withoutPruning.append(acc)
  print withPruning
  print withoutPruning
  print "average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning)


def plot(inFile):


  trainSize = range(10, 300, 25)
  withPruning = []
  withoutPruning = []
  pruningAvgs = []
  withoutPruningAvgs = []
  data = parse.parse(inFile)

  for length in trainSize:

    for i in range(100):
      random.shuffle(data)
      train = data[:int(0.9*length)]
      valid = data[int(0.9*length):length]
      test = data[length:]

      tree = ID3.ID3(train, 'democrat')
      ID3.prune(tree, valid)
      acc = ID3.test(tree, test)
      withPruning.append(acc)

      tree = ID3.ID3(train + valid, 'democrat')
      acc_noPrune = ID3.test(tree, test)
      withoutPruning.append(acc_noPrune)

    pruningAvgs.append(sum(withPruning) / len(withPruning))
    #print "ongoing"
    withoutPruningAvgs.append(sum(withoutPruning) / len(withoutPruning))
    #print "ongoing"

  fig, ax = plt.subplots()
  plt.plot(trainSize, withoutPruningAvgs, label="Without Pruning", marker='o')
  plt.plot(trainSize, pruningAvgs, label="With Pruning", marker='+')
  plt.xlabel('number of training examples')
  plt.ylabel('accuracy on test data')
  ax.set_xlim(0, 300)
  ax.set_ylim(0.5, 0.95)
  plt.legend(loc= 5)
  plt.show()


if __name__ == "__main__":
  testID3AndEvaluate()
  testPruning()
  testID3AndTest()
  testPruningOnHouseData('house_votes_84.data')
  #plot('house_votes_84.data')
