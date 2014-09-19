function [Centroids, QtdEntitiesInCluster] = i_Kmeans(Data, MinEntitiesInCluster, IsDataStandarized, k)
%First Step = Standarize data if needed
InitialSize = size(Data,1);
QtdEntitiesInCluster = [];

if IsDataStandarized == false
    r = Data - repmat(mean(Data), InitialSize ,1);
    Data = r./repmat(max(Data) - min(Data),InitialSize , 1);
end

%Second Step = Sorts Data Accordint to Distance to Zero
[~,index] = sort(sum(Data.^2,2));
Data = Data(index,:);

%Third Step Anomalous Patter
Centroids = [];
%%%%%%%%%%
iss = 1;
%%%%%%%%%%
while ~isempty (Data)
    CurrentSize = size(Data,1);
    TentCentroid = Data(CurrentSize,:); % Gets a tentative Centroid    
    DistanceToCentre = sum(Data.^2,2);
%%%%%%%%
    isp = 1;
%%%%%%%%
    while true
       % BelongsToCentroid = sum((Data-(repmat(TentCentroid, CurrentSize,1))).^2,2) < DistanceToCentre;        
        BelongsToCentroid = sum((Data-TentCentroid(ones(CurrentSize,1),:)).^2,2) < DistanceToCentre;  %faster
        NewCentroid = mean(Data(BelongsToCentroid==1,:),1);
        if isequal(TentCentroid, NewCentroid), break; end
        if (isp > 500), break; end
        TentCentroid = NewCentroid;
%%%%
        isp = isp + 1;
%%%%
    end
    
    if sum(BelongsToCentroid==1) > MinEntitiesInCluster  
        Centroids = [Centroids; NewCentroid]; %#ok<AGROW>
        QtdEntitiesInCluster = [QtdEntitiesInCluster; sum(BelongsToCentroid==1)];
    end
    Data(BelongsToCentroid==1,:)=[];

    %%%
    if(iss > 500), Data = []; end    
    iss = iss + 1;
    disp(iss);
    %%%
    
end

if nargin == 4 && size(Centroids,1) > k
    %Gets the k most populated clusters
    [~,ind] = sort(QtdEntitiesInCluster,'descend');
    Centroids = Centroids(ind(1:k),:);
end
