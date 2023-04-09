import torch

class Matrix(object):
    def __init__(self,vectors):
        self.vectors = vectors
        self.batch = vectors.size(0)
        self.dimension = vectors.size(1)
    def __str__(self):
        return self.vectors
    def plus(self,v):
        return self.vectors + v.vectors
    def minus(self,v):
        return self.vectors - v.vectors
    def magnitude(self):
        # 求 |y|
        return torch.sqrt(torch.sum(self.vectors.pow(2),dim=-1)).view(self.batch,1)
    def normalized(self):
        # y / |y|
        magnitude = self.magnitude()
        magnitude = 1.0 / magnitude
        weight = magnitude.view(self.batch,1)
        # batch x hidden
        return self.vectors * weight
    
    def component_parallel_to(self,basis):
        # 求 common vector
        u = basis.normalized()
        weight = torch.sum(self.vectors * u,dim=-1)
        weight = weight.view(self.batch,1)
        return u * weight
    
    def component_orthogonal_to(self,basis):
        # 根据 common vector 求投影
        projection = self.component_parallel_to(basis)
        # normalize basis
        # 
        return self.vectors - projection

def NB_algorithm(orignial_feature,trivial_feature):
    orignial_feature = Matrix(orignial_feature)
    trivial_feature = Matrix(trivial_feature)
    d = orignial_feature.component_orthogonal_to(trivial_feature)
    d = Matrix(d)
    f = orignial_feature.component_parallel_to(d)
    return f

