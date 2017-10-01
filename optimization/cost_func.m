%assume that reward/penalty = 1
%we fix the park configurations as 10*10 grid space
function f = cost_func(x,mu_x,sigma_x,mu_y,sigma_y)
    x = x';
    %x is 1D
    dim = size(x);
    
    sum = 0;
    normalizer = 0;
    target_prob = zeros(1,100);
    for i = 1:dim(2)
        if(rem(i,10) == 0)
            x_right = 10;
            y_bottom = i/10-1;
        else
            x_right = rem(i,10);
            y_bottom =i/10;
        end
        x_left = x_right-1;
        y_top = y_bottom+1;
        target_prob(i) = compute_target_prob(x_left,x_right,y_bottom,y_top,mu_x,sigma_x,mu_y,sigma_y);
        normalizer = normalizer + target_prob(i);
    end  
   
    for i = 1:dim(2)
        
        target_part = target_prob(i)/normalizer;
        sum = sum + (2*x(i)-1)*target_part;
    end
    f = -sum;%- for minimization
end

