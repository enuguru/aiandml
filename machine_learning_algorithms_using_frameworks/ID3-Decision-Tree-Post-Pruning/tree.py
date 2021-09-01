
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:17:35 2018

@author: sushant
"""
import pandas as pd
from random import randint
import sys
import copy
L = int(sys.argv[1])
K = int(sys.argv[2])
training_set_path = sys.argv[3]
validation_set_path = sys.argv[4]
test_set_path = sys.argv[5]
to_print = sys.argv[6]
positive = 1
negative = 0
n = 0
node_number_info = 0
node_number_variance = 0
training_data = pd.read_csv(training_set_path)
labelValues = list(training_data.columns.values)
labelValues.remove('Class')
def _unique(seq, return_counts=False, id=None):
   
    found = set()
    if id is None:
        for x in seq:
            found.add(x)
           
    else:
        for x in seq:
            x = id(x)
            if x not in found:
                found.add(x)
    found = list(found)           
    counts = [seq.count(0),seq.count(1)]
    if return_counts:
        return found,counts
    else:
        return found
     
def _sum(data):
    sum = 0
    for i in data:
        sum = sum + i
    return sum


def calculate_variance(target_values):
    values = list(target_values)
    elements,counts = _unique(values,True)
    variance_impurity = 0
    sum_counts = _sum(counts)
    for i in elements:
        variance_impurity += (-counts[i]/sum_counts*(counts[i]/sum_counts))
    return variance_impurity

def variance_impurity_gain(data, split_attribute_name, target_attribute_name):
    data_split = data.groupby(split_attribute_name)
    aggregated_data = data_split.agg({target_attribute_name : [calculate_variance, lambda x: len(x)/(len(data.index) * 1.0)] })[target_attribute_name]
    aggregated_data.columns = ['Variance', 'Observations']
    weighted_variance_impurity = sum( aggregated_data['Variance'] * aggregated_data['Observations'] )
    total_variance_impurity = calculate_variance(data[target_attribute_name])
    variance_impurity_gain = total_variance_impurity - weighted_variance_impurity
    return variance_impurity_gain

def calculate_entropy(probablities):
    import math
    sum_of_probablities = 0;
    for prob in probablities:
        sum_of_probablities += -prob*math.log(prob, 2)
    return sum_of_probablities

def calculate_entropy_of_the_list(list):
    from collections import Counter  
    cnt = Counter(x for x in list)
    num_instances = len(list)*1.0
    probs = [x / num_instances for x in cnt.values()]
    return calculate_entropy(probs)
    
def information_gain(data, split_attribute_name, target_attribute_name):
    data_split = data.groupby(split_attribute_name) 
    aggregated_data = data_split.agg({target_attribute_name : [calculate_entropy_of_the_list, lambda x: len(x)/(len(data.index) * 1.0)] })[target_attribute_name]
    aggregated_data.columns = ['Entropy', 'ProbObservations']
    new_entropy = sum( aggregated_data['Entropy'] * aggregated_data['ProbObservations'] )
    old_entropy = calculate_entropy_of_the_list(data[target_attribute_name])
    return old_entropy-new_entropy



def build_tree_using_information_gain(data, target_attribute_name, attribute_names, default_class=None):

    from collections import Counter
    count_set = Counter(x for x in data[target_attribute_name])
    global node_number_info
    if len(count_set) == 1:
        return list(count_set.keys())[0]

    elif data.empty or (not attribute_names):
        return default_class 
    
    else:
        index_of_max = list(count_set.values()).index(max(count_set.values())) 
        default_class = list(count_set.keys())[index_of_max]  
        
        info_gain = [information_gain(data, attr, target_attribute_name) for attr in attribute_names]
        index_of_max = info_gain.index(max(info_gain)) 
        best_attr = attribute_names[index_of_max]
        tree = {best_attr:{}}
        positiveCount = data['Class'].value_counts()[1];
        negativeCount = data['Class'].value_counts()[0];
        if positiveCount>negativeCount :
            best_class = 1
        elif positiveCount<negativeCount:
            best_class = 0
        else:
            best_class = 'none'
        tree[best_attr]['best_class'] = best_class
        node_number_info = node_number_info + 1
        tree[best_attr]['number'] = node_number_info
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]

        for attr_val, data_subset in data.groupby(best_attr):
            
            subtree = build_tree_using_information_gain(data_subset,
                        target_attribute_name,
                        remaining_attribute_names,
                        default_class)
            tree[best_attr][attr_val] = subtree
        return tree
def build_tree_using_variance_impurity(data, target_attribute_name, attribute_names, default_class=None):
    global node_number_variance
    from collections import Counter
    count_set = Counter(x for x in data[target_attribute_name])
    if len(count_set) == 1:
        return list(count_set.keys())[0]

    elif data.empty or (not attribute_names):
        return default_class 
    
    else:
        index_of_max = list(count_set.values()).index(max(count_set.values())) 
        default_class = list(count_set.keys())[index_of_max]
        variance_gain = [variance_impurity_gain(data, attr, target_attribute_name) for attr in attribute_names]
        index_of_max = variance_gain.index(max(variance_gain)) 
        best_attr = attribute_names[index_of_max]
        
        tree = {best_attr:{}}
        positiveCount = data['Class'].value_counts()[1];
        negativeCount = data['Class'].value_counts()[0];
        if positiveCount>negativeCount :
            best_class = 1
        elif positiveCount<negativeCount:
            best_class = 0
        else:
            best_class = 'none'
        tree[best_attr]['best_class'] = best_class
        node_number_variance = node_number_variance + 1
        tree[best_attr]['number'] = node_number_variance
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]

        for attr_val, data_subset in data.groupby(best_attr):
            subtree = build_tree_using_variance_impurity(data_subset,
                        target_attribute_name,
                        remaining_attribute_names,
                        default_class)
            tree[best_attr][attr_val] = subtree
        return tree
    
tree = build_tree_using_information_gain(training_data, 'Class', labelValues)
tree2 = build_tree_using_variance_impurity(training_data, 'Class', labelValues)


def accuracy_of_the_tree(instance, tree, default=None):
    attribute = list(tree.keys())[0]
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict): 
            return accuracy_of_the_tree(instance, result)
        else:
            return result 
    else:
        return default
    
def preorder (temptree, number):
    if isinstance(temptree, dict):
        attribute = list(temptree.keys())[0]
        if temptree[attribute]['number'] == number:
            if(temptree[attribute][0]!=0 and temptree[attribute][0]!=1):
                temp_tree = temptree[attribute][0]
                if isinstance(temp_tree, dict):
                    temp_attribute = list(temp_tree.keys())[0]
                    temptree[attribute][0] = temp_tree[temp_attribute]['best_class']
            elif(temptree[attribute][1]!=0 and temptree[attribute][1]!=1):
                temp_tree = temptree[attribute][1]
                if isinstance(temp_tree, dict):
                    temp_attribute = list(temp_tree.keys())[0]      
                    temptree[attribute][1] = temp_tree[temp_attribute]['best_class']
        else:
            left = temptree[attribute][0]
            right = temptree[attribute][1]
            preorder(left, number)
            preorder(right,number )
    return temptree



def count_number_of_non_leaf_nodes(tree):
    if isinstance(tree, dict):
        attribute = list(tree.keys())[0]
        left = tree[attribute][0]
        right = tree[attribute][1]
        return (1 + count_number_of_non_leaf_nodes(left) +  
               count_number_of_non_leaf_nodes(right)); 
    else:
        return 0;
    



def post_prune(L, K, tree):
    best_tree = tree
    for i in range(1, L+1) :
        temp_tree = copy.deepcopy(best_tree)
        M = randint(1, K);
        for j in range(1, M+1):
            n = count_number_of_non_leaf_nodes(temp_tree)
            if n> 0:
                P = randint(1,n)
            else:
                P = 0
            preorder(temp_tree, P)
        test_data['accuracyBeforePruning'] = test_data.apply(accuracy_of_the_tree, axis=1, args=(best_tree,'1') ) 
        accuracyBeforePruning = str( sum(test_data['Class']==test_data['accuracyBeforePruning'] ) / (1.0*len(test_data.index)) )
        test_data['accuracy_after_pruning'] = test_data.apply(accuracy_of_the_tree, axis=1, args=(temp_tree,'1') ) 
        accuracy_after_pruning = str( sum(test_data['Class']==test_data['accuracy_after_pruning'] ) / (1.0*len(test_data.index)) )
        if accuracy_after_pruning >= accuracyBeforePruning:
            best_tree = temp_tree
    return best_tree


test_data = pd.read_csv(test_set_path)
validation_data = pd.read_csv(validation_set_path)

if to_print == 'yes':
    print(tree)
    print(tree2)
   
test_data['predicted'] = test_data.apply(accuracy_of_the_tree, axis=1, args=(tree,'1') ) 
print( 'Accuracy with info gain ' +  (str( sum(test_data['Class']==test_data['predicted'] ) / (0.01*len(test_data.index)) )))


test_data['predicted2'] = test_data.apply(accuracy_of_the_tree, axis=1, args=(tree2,'1') ) 
print( 'Accuracy with variance ' + (str( sum(test_data['Class']==test_data['predicted2'] ) / (0.01*len(test_data.index)) )))



number_of_leaf_nodes = count_number_of_non_leaf_nodes(tree)

tree3 = post_prune(L,K,tree)
tree4 = post_prune(L,K,tree2)

test_data['predicted3'] = test_data.apply(accuracy_of_the_tree, axis=1, args=(tree3,'1') ) 
print( 'Accuracy with pruned Information gain tree ' + (str( sum(test_data['Class']==test_data['predicted3'] ) / (0.01*len(test_data.index)) )))
test_data['predicted4'] = test_data.apply(accuracy_of_the_tree, axis=1, args=(tree4,'1') ) 
print( 'Accuracy with pruned Variance gain tree ' + (str( sum(test_data['Class']==test_data['predicted4'] ) / (0.01*len(test_data.index)) )))
