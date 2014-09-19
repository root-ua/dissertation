clc;
clear;

load('datasets\accuracy.mat');
load('datasets\dataset.mat');


Datasets = [0, 10, 20, 50, 100];


Distances = ['sqEuclidean'; 'cityblock  '; 'cosine     '; 'correlation'];
Distances = cellstr(Distances);


Total = size(Datasets, 2) * size(Distances, 1) * 2;
A_ikmeans = cell(1, Total);
I = 1;

for dataset = 1:1:size(Datasets, 2)
    s_dataset = Datasets(dataset);
    
    if(I < 25)
        eval(sprintf('ZNoStd = i_Kmeans(D_F%i_NoStd, 0, true, 2);', s_dataset));
    end
    eval(sprintf('ZStd = i_Kmeans(D_F%i_Std, 0, true, 2);', s_dataset));

    for distance = 1:1:size(Distances, 1)
        s_distance = char(Distances(distance));
        
        if (I < 25)
            try
                    eval(sprintf('U = kmeans(D_F%i_NoStd, 2, ''distance'', ''%s'', ''start'', ZNoStd);', ... 
                        s_dataset, s_distance));
                    A_ikmeans{I} = {U, ZNoStd, 'noStd', s_dataset, ... 
                        s_distance, CheckLabels(U, accuracy)};
                    display(I);
                    clear U;
            catch err
            end
        end
       
        try
            eval(sprintf('U = kmeans(D_F%i_Std, 2, ''distance'', ''%s'', ''start'', ZStd);', ...
                s_dataset, s_distance));
            A_ikmeans{I+1} = {U, ZStd, 'Std', s_dataset, s_distance, ... 
                CheckLabels(U, accuracy)};
            display(I+1);
            clear U;
        catch err
        end
       
        I = I + 2;
            
    end
    
    clear ZStd ZNoStd;
end
 
save('results/ikmeans_all_std.mat', 'A_ikmeans'); 

clc;
clear;

% Final results
%   A_ikmeans{i} = [ [U], [Z], Std, Distance, Accuracy];
% U - Final Cluster Labels
% Z - Initial Centroids
