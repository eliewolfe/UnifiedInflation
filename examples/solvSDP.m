load('inflationMATLAB_.mat');

nr_unknown_moments = double(max(G(:)) - length(known_moments));
freevars = sdpvar(1, nr_unknown_moments);
lambda = sdpvar(1);
slots = [known_moments, freevars];
G = slots(G);
constraint = [ G - lambda*eye(size(G)) >= 0 ];

use_non_certificate_constraints = false;

if use_non_certificate_constraints
    for i=1:size(propto,1)
       var1 = slots(propto(i,1));
       coeff = propto(i,2);
       var2 = slots(propto(i,3));
       constraint = [constraint, var1 == coeff * var2];
    end
end

clearvars slots nr_unknown_moments;
sol = optimize(constraint,-lambda,...
    sdpsettings(...
        'solver','mosek',...
        'verbose',1,'dualize',0,...
        'showprogress',1,...
        'savesolverinput',1,'savesolveroutput',1,'debug',1)...
    );

lambdavalue = value(lambda)

fileID = fopen('result_matlab.txt','w');
fprintf(fileID,'lambda = %f\n', lambdavalue);
fclose(fileID);

if ~use_non_certificate_constraints
    X = dual(constraint);
    save dualconstraint X
end
