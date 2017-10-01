def convert_decimal(number):
		str_num = str(number)
	
		
		#REMOVE - SIGN IF ANY
		if	str_num[0]=='-':
			str_num = str_num[1:]
		
		deci_pos = str_num.index('.')
		
		
		#PROCESS THE LEFT SIDE OF THE DECIMAL
		str_deci_left = str_num[0:deci_pos]
		deci_left = int(str_deci_left)
		
		if deci_left > 0:
			return deci_left
		
		else:
			# #PROCESS THE RIGHT SIDE OF THE DECIMAL
			# #GET THE FIRST NON-ZERO DIGIT ON THE RIGHT
			deci_right = 0
			for i in range(deci_pos+1,len(str_num)):
				if int(str_num[i]) > 0:
					deci_right = int(str_num[i])
					break
				
			return deci_right

print(convert_decimal(-2.327785249317998))

