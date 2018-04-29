from sklearn.model_selection import train_test_split
import time
import copy
from Normalizer import Normalizer
from SVMClassifier import SVMClassifier

from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)
from math import log2
import random

class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
    	return str(self.name)
    def __str__(self):
    	return str(self.__class__) + ": \n" + str(self.__dict__)
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

 
  

    
t = Tree('*', [Tree('1'),
               Tree('2'),
               Tree('+', [Tree('3'),
                          Tree('4')])])




def decision_tree_learning(examples, attributes, parent_examples):

	if not examples:

		return plurality_values(parent_examples)
	elif same_classification(examples):

		return classifications(examples)
	elif not attributes:

		return plurality_values(examples)
	else:

		importance_value=importance(examples)
		exs_attribute=get_column_result(examples, importance_value)
		A= get_attribute(exs_attribute)
		tree=Tree(attributes[importance_value], [])		
		for v in A:
			temp_attr=copy.copy(attributes)
			temp_attr.pop(importance_value)
			exs=get_examples_with_attribute(examples,v,importance_value)
			if(len(temp_attr)+1==len(exs[0])):
				subtree=decision_tree_learning(exs,temp_attr,examples)
				tree.add_child(Tree(v,[]))
				tree.add_child(Tree(subtree,[]))
		return tree
		
def same_size (examples):
	size=len(examples[0])
	for i in examples:
		if(len(i)!=size):
			return False
	return True




def get_examples_with_attribute(examples,value,column):
	
	exs=[]
	examples_temp=copy.copy(examples)
	
	for i in examples_temp:


		if (i[column]==value):
			j=copy.copy(i)
			j.pop(column)
			
			exs+=[j]
			
	
	return exs




def get_attribute(results):
	attributes=[]
	for i in results:
		
		attributes.append(i[0])
	return list(set(attributes))


def importance(examples):
	
	if (examples != []):
		max_gain=0
		column=0
		
		for i in range(len(examples[0])-2):
			attribute=get_column_result(examples, i)
			gain_temp=gain(attribute)
			if(max_gain<gain_temp):
				max_gain=gain_temp
				column=i
		return column






def get_column_result(examples, column):
	list_values=[]
	for i in examples:
		value=[i[column]]+[i[len(i)-1]]
		list_values+=[value]
		
	return list_values




def entropy(q):
	sum_q=0
	for i in q:
		sum_q+=i*log2(i)
	return sum_q*-1


def remainder(attribute):
	dic={'total':len(attribute)}
	for i in attribute:
		if (dic.get(i[0])==None):
			
			dic_temp={i[len(i)-1]:1}

			dic.setdefault(i[0],dic_temp)
		else:
			
			temp_dic=dic.get(i[0])
		
			temp_value=temp_dic.get(i[len(i)-1])
			if (temp_value==None):
				dic_temp={i[len(i)-1]:1}

			else:
				dic_temp={i[len(i)-1]:temp_value+1}
			temp_dic.update(dic_temp)

	sum_entropy=0
	
	for key in list(dic.keys())[1:]:
		
		value_temp=sum(list(dic.get(key).values()))/dic.get('total')
	
		entropy_child=value_temp*entropy(list(get_prob(dic.get(key)).values()))
		sum_entropy+=entropy_child

	return sum_entropy
	

def get_count(attribute):
	dic={'total':len(attribute)}
	for i in attribute:
	
		if (dic.get(i[len(i)-1])==None):
			
			dic.setdefault(i[len(i)-1],1)
		else:
	
			temp_value=dic.get(i[len(i)-1]);
			dic_temp={i[len(i)-1]:temp_value+1}
			dic.update(dic_temp)
	return dic
def get_prob(attribute):
	dic_count=get_count(attribute)
	total_values=dic_count.get('total')
	dic_temp={}

	for key in list(dic_count.keys())[1:]:
	
		value=dic_count.get(key)
		prob= value/total_values
		dic_temp.setdefault(key,prob)
	return dic_temp

def gain(attribute):
	
	return entropy(list(get_prob(attribute).values()))-remainder(attribute)






def same_classification(examples):
	if(examples==[]):
		return False
	else:
		votes=classifications(examples)
		if(votes!=[]):
			if(len(votes)==votes.count(votes[0])):
				return True
			else:
				return False


def classifications(examples):
	list_values=[]

	for i in examples:
		size=len(i)
		list_values+=[i[size-1]]
		
	return list_values

def delete_duplicates(values):
	set_values = set(values)

	result = list(set_values)
	return result

def plurality_values(examples):
	votes=classifications(examples)

	classif=delete_duplicates(votes)
	ties=[]
	value_max=0
	classif_max=""
	for i in classif:
		temp=votes.count(i)
		if(temp>value_max):
			value_max=temp
			classif_max=i
			if(ties!=[]):
				ties=[]
		if(temp==value_max):
			ties.append(classif_max)
	if(ties==[]):
		return classif_max
	else:
		return random.choice(ties)






# Our input list.
values = [["Full","no"],["Full","no"],["Some","yes"],["Full","yes"],["Full","no"],["Some","yes"],["None","no"], ["Some","yes"],["Some","yes"],["Full","no"],["None","no"],["Full","yes"]]
values2 = [["French","yes"],["French","no"],["Italian","yes"],["Italian","no"],["Thai","yes"],["Thai","yes"],["Thai","no"],["Thai","no"],["Burger","yes"],["Burger","yes"],["Burger","no"],["Burger","no"]]
values3=[["Full","Thai","no"],["Full","French","no"],["Some","French","yes"],["Full","Thai","yes"],["Full","Italian","no"],["Some","Burger","yes"],["None","Burger","no"], ["Some","Italian","yes"],["Some","Thai","yes"],["Full","Burger","no"],["None","Thai","no"],["Full","Burger","yes"]]
lenData = 6000
samples = generar_muestra_pais(lenData)

attr=[]
for i in range(len(samples[0])-1):
	attr+=["attr"+str(i)]



print (decision_tree_learning(samples, attr,[]))


