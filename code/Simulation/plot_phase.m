%plot the phase diagram

T = readtable('dataSimulation8.csv')
res = table2array(T(:,6:end))

[argvalue, argmax] = max(res');

res2 = reshape(argmax,[20,18])
imagesc([0.5:0.2:4],[0.1:0.2:4], res2)
xlabel('\tau')
ylabel('T')
saveas(gcf,'phase8.png')