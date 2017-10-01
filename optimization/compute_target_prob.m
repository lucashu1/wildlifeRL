function f = compute_target_prob(x_left,x_right,y_bottom,y_top,mu_x,sigma_x,mu_y,sigma_y)
    alpha_l = (x_left -10*mu_x)/(10*sqrt(2)*sigma_x);
    alpha_r = (x_right -10*mu_x)/(10*sqrt(2)*sigma_x);
    beta_b = (y_bottom -10*mu_y)/(10*sqrt(2)*sigma_y);
    beta_t = (y_top -10*mu_y)/(10*sqrt(2)*sigma_y);
    
    radius_1 = sqrt(alpha_l*alpha_l + beta_b*beta_b);
    radius_2 = sqrt(alpha_r*alpha_r + beta_t*beta_t);
    
    theta_1 = atan(beta_b/alpha_l);
    theta_2 = atan(beta_t/alpha_r);
    
    f = (-1/(sqrt(2)*pi))*(theta_2-theta_1)*(exp(-radius_1*radius_1) - exp(-radius_2*radius_2));
end
    