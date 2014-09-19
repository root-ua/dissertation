clc;
clear;

load('datasets\accuracy.mat');
load('datasets\dataset.mat');


Datasets = [0, 10, 20, 50, 100];


Distances = ['sqEuclidean'; 'cityblock  '; 'cosine     '; 'correlation'];
Distances = cellstr(Distances);


Total = size(Datasets, 2) * size(Distances, 1) * 2;

A_kmeans = cell(1, Total);
Errors = cell(1, Total);
I = 1;

for dataset = 1:1:size(Datasets, 2)
    s_dataset = Datasets(dataset);

    for distance = 1:1:size(Distances, 1)
        s_distance = char(Distances(distance));
            
        try
            eval(sprintf('U = kmeans(D_F%i_Std, 2, ''replicates'', 100, ''distance'', ''%s'');', ...
                s_dataset, s_distance));
            A_kmeans{I} = {U, 'Std', s_dataset, s_distance, ...
                CheckLabels(U, accuracy)};
            display(I);
        catch err
            Errors{I} = {ex};
             display(ex);
        end
        
        try
            eval(sprintf('U = kmeans(D_F%i_NoStd, 2, ''replicates'', 100, ''distance'', ''%s'');', ... 
            s_dataset, s_distance));
            A_kmeans{I} = {U, 'noStd', s_dataset, s_distance, ...
                CheckLabels(U, accuracy)};
            display(I);
        catch err
            Errors{I} = {err};
            display(err);
        end
            
    I = I + 2;
    
    end
      
end
 
save('results/kmeans_all_nostd_all.mat', 'A_kmeans', 'Errors'); 

clc;
clear;

% Final results
%   A_kmeans{i} = [ [U], Std, Distance, Accuracy];
