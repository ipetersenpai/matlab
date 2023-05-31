close all
clear all

load fisheriris;
trainData = meas(:, 3:4);
trainLabel = species;

Y = categorical (species);
nbClass = categories(Y);
tabulate(Y);

x = trainData(:,1);
y = trainData(:,2);
gscatter(x,y);
gscatter(x,y,trainLabel);
figure, gscatter(x,y, trainLabel, 'rgb','osd');
xlabel('Petal length');
ylabel ('Petal width');

tree = ClassificationTree.fit(trainData,trainLabel);


view(tree,'mode','graph');

testData = trainData;
testLabel = species;
dtClass = predict(tree, testData);

trainData = trainData;
testLabel = species;

[dtClass, score] = predict(tree,testData);
dtAccuracy = compAccuracy(dtClass,testLabel)*100;

%create model using NB
Mdl = fitcnb(trainData,trainLabel);

[nbClass,Posterior,Cost] = predict(Mdl,testData);

acc = compAccuracy(nbClass, testLabel);
nbAccuracy = acc*100;

message = sprintf('Classifier''s accuracy: \n DT = %.2f%% \n NB = %.2f%%'...
                    ,dtAccuracy, nbAccuracy);
msgbox(message,'modal');


