function acc = compAccuracy(class_val, testLabel)

samples = size(testLabel, 1);
bad = ~strcmp(class_val,testLabel);
wrong_class = sum(bad);
acc = (samples-wrong_class)/samples;

end

