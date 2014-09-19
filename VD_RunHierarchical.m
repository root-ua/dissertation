clc;
clear;

load('datasets\accuracy.mat');
load('datasets\dataset.mat');

Datasets = [0, 10, 20, 50, 100];

Methods = ['average '; 'centroid'; 'complete'; 'median  '; 
    'single  '; 'ward    '; 'weighted'];
Methods = cellstr(Methods);

Distances = ['euclidean  '; 'seuclidean '; 'cityblock  '; 
    'minkowski  '; 'chebychev  '; 'mahalanobis'; 'cosine     '; 
    'correlation'; 'spearman   '; 'hamming    '; 'jaccard    '];
Distances = cellstr(Distances);

Total = size(Datasets, 2) * size(Methods, 1) * size(Distances, 1) * 2;
A_Hierarchical = cell(1, Total);
Errors = cell(1, Total);
I = 1;

for dataset = 1:1:size(Datasets, 2)
   s_dataset = Datasets(dataset);
	
    for method = 1:1:size(Methods, 1)
        s_method = char(Methods(method));
        
        for distance = 1:1:size(Distances, 1)
            s_distance = char(Distances(distance));
     
            try
                eval(sprintf('Z = linkage(D_F%i_Std, ''%s'', ''%s'');', ... 
                    s_dataset, s_method, s_distance));
                U = cluster(Z, 'maxclust', 2);
                A_Hierarchical{I} = {U, Z, 'Std', s_dataset, s_method, ...
                    s_distance, CheckLabels(U, accuracy)};
                clear U Z;
                disp(I);
            catch ex
                Errors{I} = {ex, 'Std', s_dataset, s_method, s_distance};
                disp(ex);
            end
            try 
                eval(sprintf('Z = linkage(D_F%i_NoStd, ''%s'', ''%s'');', ... 
                s_dataset, s_method, s_distance));
                U = cluster(Z, 'maxclust', 2);
                A_Hierarchical{I+1} = {U, Z, 'NoStd', s_dataset, s_method, ...
                    s_distance, CheckLabels(U, accuracy)};
                clear U Z;
                disp(I+1);
            catch ex
                Errors{I+1} = {ex, 'NoStd', s_dataset, s_method, s_distance};
                disp(ex);
            end
           
            I = I + 2;
            
        end
    end
end

save('results/Hierarchical_all.mat', 'A_Hierarchical', 'Errors');
save('results/Hierarchical_Errors.mat', 'A_Hierarchical', 'Errors');

clear;
clc;