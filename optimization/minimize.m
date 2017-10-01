function minimize(mu_x,sigma_x,mu_y,sigma_y)
    x = zeros(100,1); %coverage probabilities
    x0 = zeros(100,1);
    x0(1,:) = 0.5;
    %create the min bounds for the weight vector
    min = zeros(100,1);
    min(1,:) = 0;
    %create the max bounds for the weight vector
    max = zeros(100,1);
    max(1,:) = 1;
    A = zeros(1,100);
    A(1:100) = 1;
    B = zeros(1,1);
    B(1,1) = 1;
    %set the options parameter with the algorithm used by fmincon
    options = optimset('Algorithm','interior-point','TolFun',1e-8,'Display','notify');
    
    [coverage, avg_error, exitflag] = fmincon(@(x)cost_func(x,mu_x,sigma_x,mu_y,sigma_y),x0,[],[],A,B,min,max,[],options);
    
    coverage
    avg_error
end