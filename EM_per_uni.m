%This script: (cf. paper "Improving on Yelp Reviews Using NLP and Bayesian Scoring")
%	Reads the CSV file containing the list of schools represented in Yelp's Academic Dataset
%	For each school, trains a Bayesian model of nearby businesses and related reviewers, 
%These models can then be used to produce a plot of all universities depending on their average intrinsic business value and average reviewer bias, 
nb_ini = 3;
max_it =200;

schools = importdata('school_list.csv');
nb_School = length(schools);
AvgmU = zeros(nb_School,1);
AvgmB = zeros(nb_School,1);

for k = 1:length(schools)
    
     disp(['Learning model for',schools(k)]) 
    
     G = spconvert(importdata(['reviews',num2str(k),'.csv']));
    %1 raw = 1 business / 1 column = 1 user
    G=G(:,sum(G~=0,1)~=0); %Get rid of useless users, or B will have zero->NaN
    [nB,nU] = size(G);
    U = full(sum(G~=0,2)); %Number of users per business
    B = full(sum(G~=0,1)); %Number of businesses per user
    N = sum(B); %Total number of reviews
    sparsity = G~=0;
        
    %For each school, train nb_ini models and take retain the one with
    %higher likelihood to output the required quantities
    
    model_like = zeros(nb_ini,1);
    model_AvgmU = zeros(nb_ini,1);
    model_AvgmB = zeros(nb_ini,1);
    
    %Regularization and stopping criteria

    alpha=2;
    beta=1;
    epsilon = 1e-3;
        
    
    for ini=1:nb_ini

        %Initialization

        s=1+rand(1);
        mB = 5 * rand(nB,1);
        sB = 1+5*rand(nB,1);
        mU = 5* rand(1,nU);
        sU = 1+5*rand(1,nU);
        L=[-Inf];

        for it=1:max_it

        %E-step

        %Precomputations

        mBmat= repmat(mB,1,nU); 
        sBmat= repmat(sB,1,nU);
        mUmat= repmat(mU,nB,1); 
        sUmat= repmat(sU,nB,1);
        norm = real(1./bsxfun(@plus,sU,sB+s));
        invsBmat = real(1./sBmat);
        invsUmat = real(1./sUmat);

        %Sparsification

        mBmat= sparsity.*mBmat;
        sBmat= sparsity.*sBmat;
        mUmat= sparsity.*mUmat;
        sUmat= sparsity.*sUmat;
        norm = sparsity.*norm;
        invsBmat = sparsity.*invsBmat;
        invsUmat = sparsity.*invsUmat;

        %E-Step

        mYZ_XB = real((mBmat.*(sUmat + s) + sBmat.*(G- mUmat)).*norm);
        mYZ_XU = real((mUmat.*(sBmat + s) + sUmat.*(G- mBmat)).*norm);

        sYZ_XBB = real(sBmat.*(sUmat + s).*norm);
        sYZ_XUU = real(sUmat.*(sBmat +s).*norm);
        sYZ_XBU = real(- sBmat.*sUmat.*norm);


        %Avoid numerical problems:

        mYZ_XB(abs(mYZ_XB)<1e-10) = 0;
        mYZ_XU(abs(mYZ_XU)<1e-10) = 0;
        sYZ_XBB(sYZ_XBB<1e-10) = 0;
        sYZ_XUU(sYZ_XUU<1e-10) = 0;
        sYZ_XBU(abs(sYZ_XBU)<1e-10) = 0;

        %Computation of the likelihood

        l = real(-1/(2*s)*sum(sum((G - mYZ_XB - mYZ_XU).^2))...
            -1/(2*s)*[1,1]*([sum(sum(sYZ_XBB)),sum(sum(sYZ_XBU));sum(sum(sYZ_XUU)),sum(sum(sYZ_XBU))])*[1;1]...
            -1/2* sum(sum(((mBmat - mYZ_XB).^2 + sYZ_XBB).*invsBmat))...
            -1/2* sum(sum(((mUmat - mYZ_XU).^2 + sYZ_XUU).*invsBmat))...
            -N*log(s) - U'*log(sB) - B*log(sU)' ...
            -1/2 *sum(sum(spfun(@log,sYZ_XBB.*sYZ_XUU - sYZ_XBU.^2)))...
            -(alpha+1)*sum(log(sB))-sum(beta./sB) -(alpha+1)*sum(log(sU))-sum(beta./sU) -(alpha+1)*log(s)-beta./s);

        L= [L,l] ;
        if abs(L(it+1)-L(it))/abs(L(it+1)) < epsilon
            break
        end
        %M-step

        NEWmB = real(sum(mYZ_XB,2)./U);
        NEWsB = real(2*beta+sum((mBmat-mYZ_XB).^2,2) + sum(sYZ_XBB,2))./(U+2*(alpha+1));
        NEWmU = real(sum(mYZ_XU,1)./B);
        NEWsU = real(2*beta+sum((mUmat-mYZ_XU).^2,1) + sum(sYZ_XUU,1))./(B+2*(alpha+1));
        NEWs = real((2*beta+sum(sum((G - mYZ_XB - mYZ_XU).^2)) + [1,1]*([sum(sum(sYZ_XBB)),sum(sum(sYZ_XBU));sum(sum(sYZ_XUU)),sum(sum(sYZ_XBU))])*[1;1])/(N+2*(alpha+1)));

        %Avoid numerical problems:

        NEWmB(abs(NEWmB)<1e-10) = 0;
        NEWsB(NEWsB<1e-10) = 0;
        NEWmU(abs(NEWmU)<1e-10) = 0;
        NEWsU(NEWsU<1e-10) = 0;
        NEWs(NEWs<1e-10) = 0;

        if (any(any(isnan(mBmat))) || any(any(isnan(sBmat))) || any(any(isnan(mUmat))) ...
                        || any(any(isnan(sUmat))) || any(any(isnan(norm))) || any(any(isnan(mYZ_XB))) ...
                        || any(any(isnan(mYZ_XU))) || any(any(isnan(sYZ_XBB))) || any(any(isnan(sYZ_XUU)))...
                        || any(any(isnan(sYZ_XBU))) ||any(any(isnan(NEWmB)))||any(any(isnan(NEWsB)))...
                        ||any(any(isnan(NEWmU)))||any(any(isnan(NEWsU)))||isnan(NEWs)),
                    disp('NAN!')
                    break;
        end
        mb=NEWmB;
        sB=NEWsB;
        mU=NEWmU;
        sU=NEWsU;
        s=NEWs;
        end
        
        model_like(ini) = L(end);
        model_AvgmU(ini) = mean(mU);
        model_AvgmB(ini) = mean(mB);
    end
    [~,best_ini] = max(model_like);
    AvgmU(k) = model_AvgmU(best_ini);
    AvgmB(k) = model_AvgmB(best_ini);
end


% dx = 0.01; dy = 0.01;
% plot(AvgmB,AvgmU,'.')
% text(AvgmB+dx, AvgmU+dy, schools);
