from sklearn.model_selection import train_test_split
import time
import copy
from Normalizer import Normalizer
from SVMClassifier import SVMClassifier

from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)
from math import log2, sqrt
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
import operator
sys.setrecursionlimit(9999999)
class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', children=None):
    
        self.name = name
        self.children = []
        self.gain=0
        self.votes= None
      
        self.votes_test_result= None
      

        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
    	return str(self.name)
    def __str__(self):

    	return str(self.__class__) + ": \n" + str(self.__dict__)
    def set_data_pruning(self,p,n):
    	#print("POSITIVOS:"+str(p))
    	self.p+=p
    	self.n+=n
    	self.set_size=self.p+self.n
    	self.set_error=(self.set_size-self.n)/self.set_size*100


    def add_child(self, node):
    	
    	assert isinstance(node, Tree)
    	self.children.append(node)
    def set_gain(self,gain):
    	self.gain=gain
    def get_name(self):
    	return self.name
    def get_children(self):
    	return self.children
    def num_children(self):
       	return len(self.children)
    def get_child(self, name):
    	
    	for i in self.children:

    		if(name==i.get_name()):
    			
    			return i
    	
    	
    	return None 
    def delete_child(self, child):
    	self.children.remove(child)
    def get_leafs(self):
    	
    	votes=[]
    	if(self.votes!=None):
	    	for i in self.votes.keys():
	    		
	    		votes+=[[i]]

    	return Tree(votes,[])


	    	
    def printTree(self):

    	if(self.children!=[]):
    		print("---------------------------------------------------------------------")

    		print(self)
    		for i in self.children:
    			i.printTree()
    	else:
    		print(self)
    		print("---------------------------------------------------------------------")



    def calc_pk_nk(self):
    	if(self.children!=[]):
   		
    		for i in self.children:

    			if(self.p + self.n>0):
	    			i.pk = self.p*((i.p+i.n)/(self.p+self.n))
	    			i.nk = self.n*((i.p+i.n)/(self.p+self.n))

	    			i.calc_pk_nk()

    def total_votes(self):
    	tot=0
    	for i in self.votes.keys():
    		tot+=self.votes.get(i)
    	return tot 


    def test(self, test_list, attributes, expected):
    	dic={}
    	
  
    	if(self.name in attributes):
    		

    		position_attr=attributes.index(self.name)
    		attr=test_list[position_attr]
    		child=self.get_child(attr)
    		valor=['NULO']
    		if(child!=None):

    			
    			
	    		for i in child.children:
	    		
	    			valor=i.test(test_list,attributes,expected)
	    	
	    			if(i.children==[]):
	    			
	    				self.add_votes_test_result(i.name)
	    				i.add_votes_test_result(i.name)
	    				child.add_votes_test_result(i.name)
	    				return i.name

	    		child.add_votes_test_result(valor)
	    		self.add_votes_test_result(valor)
	    		return valor
	    	else:
	    		self.add_votes_test_result(['NULO'])
	    		return ['NULO']
    	else:
    		

    		name=self.name
    		self.add_votes_test_result(name)
    		return self.name

    		

 
    	


    def add_votes_test_result(self,name):
    	
    	if(isinstance(name,list)):
    		name=name[0]
    	
    		if(isinstance(name,list)):
    			name=name[0]

    

    	if(self.votes_test_result!=None):
    		count_values=self.votes_test_result.get(name)
    		if(count_values!=None):
    			self.votes_test_result[name]=count_values+1
    		else:
    			self.votes_test_result[name]=1
    	else:
    		self.votes_test_result={}
    		#print(name)
    		self.votes_test_result[name]=1


    def observed_table(self,results):
    	table=[]
    	for i in self.children:
    		row=[]
    		for j in results:
    
    			if(i.votes_test_result!=None):
	    			if(i.votes_test_result.get(j)!=None):
	    				row+=[i.votes_test_result.get(j)]
	    			else:
	    				row+=[0]
    		table+=[row]
    		row=[]
    	return table 
    def expected_table(self, table):
    	table_expected=[]
    	total_table=self.total_table(table)
    	#print(total_table)
    	for i in table:
    		temp=[]
    		for j in range(len(i)):
    			column_total=self.total_column_observed_table(table, j)
    			row_total=sum(i)
    			if(total_table>0):
    				temp+=[column_total*row_total/total_table]
    			else:
    				temp+=[0]

    		table_expected+=[temp]
    	return table_expected

    def chi_square(self, expected_table,observed_table):
    	chi_sum=0
    	for i in range(len(observed_table)):

    		for j in range(len(observed_table[i])):
    			if(expected_table[i][j]!=0):
    				chi=(observed_table[i][j]-expected_table[i][j])**2/expected_table[i][j]
    				chi_sum+=chi

    	return chi_sum






    def total_table(self,table):
    	total=0
    	for i in table:
    		total+=sum(i)
    	return total


    def total_column_observed_table(self, table, column):
    	sum_column=0

    	for i in table:

    		if(i!=[]):
    			sum_column+= i[column]

    	return sum_column


    #def chi_square(self):

    def desviation(self,data):
    	desv=0

    	for i in self.children:
    		for j in data:
    			n_k=0
    			if(i.votes_test_result!=None):
    				if(i.votes_test_result.get(j)!=None and self.votes_test_result.get(j)!=None):
    					
    					
    					p=self.votes_test_result.get(j)
    					n= sum_votes(self.votes_test_result)-p
    					
    					pk=i.votes_test_result.get(j)
    					nk=sum_votes(i.votes_test_result)-pk

    					pl=(pk+nk)/(p+n)
    					n_k=p*pl
    					desv+=(n_k-pk)**2/n_k
    					
    					
    	print(desv)	
    	return  desv

    def pruning_chi(self, threshold, attr, data):


    	if(self.children!=[]):
    		for i in self.children:
    			t_obs=i.observed_table(data)
    			t_expc=i.expected_table(t_obs)
    			if(t_obs!=[[]] and 0<len(t_expc)):
    				i.pruning_chi(threshold,attr, data)
    				
    				chip=self.desviation(data)
    				chih=i.desviation(data)
    				if(chip<threshold and chih<threshold):
    					self.children=[]
    					#self.delete_child(i)
    					self.add_child(i.get_leafs())

    def pruning_chi2(self, threshold, attr, data):
    	if (self.children==[]):
    		print(self.desviation(data))
    		return self.desviation(data)
    	else:
    		for i in self.children:
    			n=i.pruning_chi2(threshold,attr,data)
    			p=self.desviation(data)
    			print("n"+str(n))
    			print("p"+str(p))
    			if(n<threshold and p<threshold):
    				dic=self.votes
    				max_value=max(dic.values());
    				for key in dic.keys():
    					if(dic[key] == max_value):
    						max_key = key
    						break;
    				self.children=[Tree(max_key,[])]
    				break;

    		return p
    			
    	

def sum_votes(votes):
	val=0
	for i in votes.keys():
		val+=votes.get(i)
	return val

  





def decision_tree_learning(examples, attributes, parent_examples):
	
	if not examples:
		
		return  plurality_values(parent_examples)
	elif same_classification(examples):
		
		return classifications(examples)
	elif not attributes:
		
		return  plurality_values(examples)
	else:

		importance_value , max_gain=importance(examples)
		exs_attribute=get_column_result(examples, importance_value)
		A= get_attribute(exs_attribute)
		tree=Tree(attributes[importance_value], [])	
		tree.set_gain(max_gain)	
		tree.votes=count_votes(examples)
		for v in A:
			
			temp_attr=copy.copy(attributes)
			temp_attr.pop(importance_value)
			exs=get_examples_with_attribute(examples,v,importance_value)

			if(len(temp_attr)+1==len(exs[0])):

				subtree=decision_tree_learning(exs,temp_attr,examples)
				
				if(isinstance(subtree, Tree)):
					
					temp_tree=Tree(v,[subtree])
				else:
					
				
					temp_tree=Tree(v,[Tree(subtree,[])])
				temp_tree.votes=count_votes(exs)
				tree.add_child(temp_tree)
				
				


				

		
		return tree
def count_votes(examples):
	temp_list=[]
	for i in examples:
		temp_list+=[i[len(i)-1]]
	dic={}
	for i in list(set(temp_list)):
		
		dic.setdefault(i,temp_list.count(i))



	return dic
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
		
		return [column,max_gain]






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
		
	#print(delete_duplicates(list_values))
	return  delete_duplicates(list_values)

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
data=[]
for i in samples:
	data+=[i[len(i)-1]]
data=list(set(data))



lenData1 = 3000
samples_pruning = generar_muestra_pais(lenData1)

attr=[]
for i in range(len(samples[0])-1):
	attr+=["attr"+str(i)]


tree_test=decision_tree_learning(samples, attr,[])


#tree_test.test(samples[3],attr,samples[3][len(samples[3])-1])





fail=0
win=0
for i in samples_pruning:

	
	result=tree_test.test(i,attr,i[len(i)-1])


	if(result[0]==i[len(i)-1]):
		win+=1
		
	else:
		fail+=1
		
print("---------------------------------------------------------------------")		
print("---------------------------------------------------------------------")
print("SIN PODADO")
print("---------------------------------------------------------------------")
print("ACERTADOS:"+str(win))
print("FALLADOS: "+str(fail))
print("ACCURACY: "+str((len(samples_pruning)-fail)/len(samples_pruning)*100))
print("---------------------------------------------------------------------")

print(tree_test)
#print(tree_test.get_leafs())
#print(tree_test.desviation(data))
#print(tree_test.chi_square(tree_test.expected_table(tree_test.observed_table(data)),tree_test.observed_table(data)))
tree_test.pruning_chi2(0.05,attr,data)
'''
print(tree_test.observed_table(data))
print(tree_test.expected_table(tree_test.observed_table(data)))
print(tree_test.children[0])

print(tree_test.children[0].children[0].children[0])
print(tree_test.children[0].children[0].children[0].observed_table(data))
print(tree_test.children[0].children[0].children[0].expected_table(tree_test.observed_table(data)))


#tree_test.calc_pk_nk()
#print(tree_test.observed_table(data))
#print(tree_test.expected_table(tree_test.observed_table(data)))
print(tree_test.chi_square(tree_test.expected_table(tree_test.observed_table(data)),tree_test.observed_table(data)))
print(tree_test.children[0].desviation(data,tree_test.votes_test_result))

tree_test.pruning_chi(0.5,attr,data,None)
'''

fail=0
win=0
for i in samples_pruning:

	result=tree_test.test(i,attr,i[len(i)-1])


	if(result[0]==i[len(i)-1]):
		win+=1
		
	else:
		fail+=1
print("---------------------------------------------------------------------")
print("PODADO")
print("---------------------------------------------------------------------")
print("ACERTADOS:"+str(win))
print("FALLADOS: "+str(fail))
print("ACCURACY: "+str((len(samples_pruning)-fail)/len(samples_pruning)*100))
print("---------------------------------------------------------------------")

