import numpy as np
import scipy.stats as stats

class DPM(object):
	'''
	Class to perform semi-parametric inference for multivariate data,
	implemented using blocked Gibbs sampler.

	'''
	def __init__(self,prior_dict,kernel='mvn',data=None,data_file='../hwk2-data.txt',n=0,etol=0.001,indv_phi=False):
	
		if data is None:
			self.data = np.loadtxt(data_file)
		else:
			self.data = data

		try:
			self.n_dim = self.data[0,:].size
		except:
			self.data = self.data.reshape(self.data[:].size,1)
   
		self.n_dim = self.data[0,:].size            
		self.n_data = self.data[:,0].size

		#check if we are pooling phi, or picking individual phi for each point
		self.indv_phi = indv_phi

		#set the truncation value
		if n < 1:
			self.n=100
		else:
			self.n = n
   
		self.pos={}
		#set the hyperpriors
		self.set_hyperpriors(prior_dict)

		#set the initial values
		self.set_pos()

		#set create the hodling arrays
		self.init_arrays()
		
		
	def mcmc_step(self,n_step=1):
		'''
		Run an iteration of the MCMC sampler. 
		'''
		for i in range(n_step):
			self.update_z()
			self.update_p()
			self.update_l()
			
			self.z_arr = np.dstack([self.z_arr,self.pos['z'].reshape(self.n,self.n_dim,1)])
			self.l_arr = np.vstack([self.l_arr,self.pos['l']])
			self.p_arr = np.vstack([self.p_arr,self.pos['p']])
			self.n_star_arr = np.hstack([self.n_star_arr,np.unique(self.pos['l']).size])


			#if self.update_phi:
			#	self.update_phi()
			#	self.phi_arr = np.append(self.phi_arr,self.pos['phi'])
			if self.update_phi and self.indv_phi:
				self.draw_phi_indv()
				if self.n_dim > 1:
					self.phi_arr = np.concatenate((phi_arr[...,np.newaxis],self.pos['phi'][...,np.newaxis]),axis=-1)
				else:
					self.phi_arr = np.dstack([self.phi_arr,self.pos['phi']])
				
			elif self.update_phi:
				self.draw_phi()
				if self.n_dim > 1:
					self.phi_arr = np.dstack([self.phi_arr,self.pos['phi']])
				else:
					self.phi_arr = np.append(self.phi_arr,self.pos['phi'])

			if self.update_alpha:
				self.draw_alpha()
				self.alpha_arr = np.append(self.alpha_arr,self.pos['alpha'])
				
			if self.update_psi_loc:
				self.draw_psi_loc()
				if self.n_dim > 1:
					self.psi_loc_arr = np.vstack([self.psi_loc_arr,self.pos['psi_loc']])
				else:
					self.psi_loc_arr = np.append(self.psi_loc_arr,self.pos['psi_loc'])
			
			if self.update_psi_cov:
				self.draw_psi_cov()
				self.psi_cov_arr = np.append(self.psi_cov_arr,self.pos['psi_cov'])

				        
	def set_pos(self):
		'''
		Set the initial values for the pos array.
		'''
		self.pos['z'] = np.zeros((self.n,self.n_dim))
		self.pos['l'] = np.zeros(self.n_data,dtype=int)
		self.pos['p'] = np.zeros(self.n)



	def init_arrays(self):
		'''
		Create the arrays to hold the MCMC iterations.
		'''
		self.z_arr = self.pos['z'].reshape(self.n,self.n_dim,1)
		self.l_arr = self.pos['l']
		self.p_arr = self.pos['p']
		self.alpha_arr = self.pos['alpha']
		self.n_star_arr = np.unique(self.pos['l']).size
		self.phi_arr = self.pos['phi']
		self.psi_loc_arr = self.pos['psi_loc']
		self.psi_cov_arr = self.pos['psi_cov']
		
	def set_hyperpriors(self,prior_dict):
		'''
		Set the hyperpriors for the sampling.
		'''
	    #if alpha exists, fix, alpha
		if 'alpha' in prior_dict:
			self.pos['alpha'] = prior_dict['alpha']
			self.update_alpha = False
			print 'Alpha fixed to ' + str(self.pos['alpha'])
		else:
			try:
				self.a_alpha = prior_dict['a_alpha']
				self.b_alpha = prior_dict['b_alpha']
				self.pos['alpha'] = self.a_alpha * self.b_alpha
				self.update_alpha = True
			except KeyError:
				print 'Alpha not fixed and no hyperpriors for Alpha provided'
				raise

		#if psi_loc exists, fix psi_loc
		if 'psi_loc' in prior_dict:
			self.pos['psi_loc'] = prior_dict['psi_loc']
			print 'Psi_loc fixed to ' + str(self.pos['psi_loc'])
			self.update_psi_loc = False
		else:
			try:
				self.mean_psi_loc = prior_dict['mean_psi_loc']
				self.cov_psi_loc = prior_dict['cov_psi_loc']
				self.pos['psi_loc'] = self.mean_psi_loc
				self.update_psi_loc = True
			except KeyError:
				print 'Psi not fixed and no hyperpriors for Psi_loc provided'
				raise

		#if psi_cov exists, fix psi_scale
		if 'psi_cov' in prior_dict:
			self.pos['psi_cov'] = prior_dict['psi_cov']
			print 'Psi_cov fixed to ' + str(self.pos['psi_cov'])
			self.update_psi_cov = False
		else:
			try:
				self.nu_psi_cov = prior_dict['nu_psi_cov']
				self.cov_psi_cov = prior_dict['cov_psi_cov']
				
				if self.n_dim > 1:
					self.pos['psi_cov'] = self.cov_psi_cov / (self.nu_psi_cov - self.cov_psi_cov.shape[0] - 1.)
				else:
					self.pos['psi_cov'] = self.cov_psi_cov / (self.nu_psi_cov - 1)
					
				self.update_psi_cov = True
			except KeyError:
				print 'Psi not fixed and no hyperpriors for psi_cov provided'
				raise
				
		#if phi exists, fix phi
		if 'phi' in prior_dict:
			self.pos['phi'] = prior_dict['phi']
			self.update_phi = False
			print 'Phi fixed to ' + str(self.pos['phi'])
		else:
			try:
				self.nu_phi = prior_dict['nu_phi']
				self.cov_phi = prior_dict['cov_phi']
				if self.n_dim > 1:
					self.pos['phi'] = self.cov_phi / (self.nu_phi - self.n_dim - 1.)
				else:
					self.pos['phi'] = self.cov_phi / (self.nu_phi - 1)
				self.update_phi = True
			except KeyError:
				print 'Phi not fixed and no hyperpriors for Phi provided'
				raise
				
		if self.indv_phi:
			self.pos['phi'] = np.tile(self.pos['phi'],(self.n,1)).reshape(self.n,self.n_dim,self.n_dim)
        
        
	def update_z(self):
		'''
		Update the z values, one by one.  
		'''
		#extract the necessary values for z updates
		z = self.pos['z']
		z_new = np.zeros(z.shape)
		l = self.pos['l']
		phi = self.pos['phi']
		psi_loc = self.pos['psi_loc']
		psi_cov = self.pos['psi_cov']
		l_star = np.unique(l)
		z_unmatched = np.delete(z,l_star)
		
		#compute inverses for psi covariance matrix:
		if self.n_dim > 1:
			psi_cov_inv = np.linalg.inv(psi_cov)
		else:
			psi_cov_inv = 1./psi_cov
			
			
		#if not doing individual phis, can just
		#compute phi inverse matrixes once
		if self.n_dim and not self.indv_phi:
			phi_inv = np.linalg.inv(phi)
			
		elif not self.indv_phi:
			phi_inv = 1./phi
		
		 
		#need to sample new z values one by one
		for i in range(z_new.shape[0]):
			
			if i in l_star:
				#if not pooling phi, then we need to calculate
				#each phi inverse individually
				if self.n_dim > 1 and self.indv_phi:
					phi_inv = np.linalg.inv(phi[i,...])
				elif self.indv_phi:
					phi_inv = 1./phi[i,...]
				inds = (l == i)
				data_star = self.data[inds,:]

				#if multidimensional, need to compute inverses
				#of covariance matrices
				if self.n_dim > 1:
					this_z = z[i,:]
					self.this_z  = this_z 
					this_cov = np.linalg.inv(psi_cov_inv + data_star[:,0].size * phi_inv)
					self.this_cov = this_cov
					self.phi_inv = phi_inv
					self.data_star = data_star
					this_mean = np.dot(this_cov,\
				        (np.dot(psi_cov_inv,this_z) + data_star[:,0].size * np.dot(phi_inv,np.mean(data_star[:,:],axis=0))))
					
					self.this_cov = this_cov
					self.this_mean = this_mean
					
					z_new[i,:] = stats.multivariate_normal.rvs(
						mean=this_mean,cov=this_cov)
                   

				else:
					this_z = z[i,0]
					this_cov = 1./(psi_cov_inv + data_star.size * phi_inv)
					this_mean = this_cov *\
					    (psi_cov_inv*this_z + data_star[:].size * phi_inv* np.mean(data_star[:]))
					z_new[i,:] = stats.norm.rvs(loc=this_mean,scale=np.sqrt(this_cov))

				self.this_cov = this_cov
				self.this_mean = this_mean


			else:
				#if z isn't matched to an l, draw a new value from the prior.
				if self.n_dim > 1:
					z_new[i,:] = stats.multivariate_normal.rvs(
						mean=psi_loc,cov=psi_cov)
						
				else:
					z_new[i,:] = stats.norm.rvs(loc=psi_loc,scale=psi_cov)
        
		self.pos['z'] = z_new
        
	def update_p(self):
		'''
		Update the probability array using stick-breaking construction.
		'''
		#extract the relevant values from the pos array
		alpha = self.pos['alpha']
		l = self.pos['l']		
		l_star,l_counts = np.unique(l,return_counts=True)
		
		#create blank p and m_l arrays to hold values
		p = np.zeros(self.n)
		m_l = np.zeros(self.n)
		
		#populate the m_l array with the non-zero values
		m_l[l_star] = l_counts
		
		self.m_l = m_l
		
		#reverse the m_l and sum for the b-term in the beta RV generator
		m_l_rev_sum = np.cumsum(m_l[::-1])[::-1]
		
		self.m_l_rev_sum = m_l_rev_sum
		
		#generate beta draws, v_l_star is one ind shorter than p
		v_l_star = stats.beta.rvs(1 + m_l[:-1],alpha + m_l_rev_sum[1:])

		self.v_l_star = v_l_star
		
		#stick breaking construction for p-array creation
		p[0] = v_l_star[0]
		p[1:-1] = v_l_star[1:] * np.exp(np.cumsum(np.log(1.-v_l_star[0:-1])))  
		
		p[-1] = 1.-np.sum(p[:-1])
		
		self.p = p

		self.pos['p'] = p
		self.pos['v_l_star'] = v_l_star
        
	def update_l(self):
		'''
		Update the values for l. Probably needs to be done one-by-one.
		
		Parameters can be explicitly passed for resampling purposes.
		'''
		p = self.pos['p']
		phi = self.pos['phi']
		z = self.pos['z']
		l = self.pos['l']
		
		for i in range(l.size):
			if self.n_dim > 1 and not self.indv_phi:
				probs = p * stats.multivariate_normal.pdf(self.data[i,:],mean=z[:],cov=phi)
				
			elif self.n_dim > 1 and self.indv_phi:
				l_pdf = np.zeros(self.n)
				for j in range(self.n):
					l_pdf[j] = stats.multivariate_normal.logpdf(self.data[i,:],mean=z[j,...],cov=phi[j,:,:])
				probs = p * np.exp(np.sum(l_pdf))
				
			else:
				probs = p * stats.norm.pdf(self.data[i,:],loc=z[:,0],scale=np.sqrt(phi))
				
			probs[probs < 1e-10] = 0.
			probs = probs / np.sum(probs)
				
			self.probs = probs
			l[i] = np.where(np.random.multinomial(1,probs) != 0)[0][0]
			
		self.pos['l'] = l
        
	def draw_phi(self):
		'''
		Update values for phi, when phi assumed to be constant across kernels. 
		'''
		l = self.pos['l']
		z = self.pos['z']
 		
		if self.n_dim > 1:
			scale_inv = np.linalg.inv(self.cov_phi)
		else:
			scale_inv = 1./self.cov_phi

		nu_star = self.nu_phi + l.size
		
		if self.n_dim == 1:
			z_l = z[l,0]
			scale_star = self.cov_phi + np.sum((self.data[:,0] - z_l)**2)
		else:
			z_l = z[l,:]
			scale_star = self.cov_phi + np.sum((self.data[:] - z_l)**2)               

		self.pos['phi'] = stats.wishart.rvs(df=nu_star,scale=scale_star)
		
	def draw_phi_indv(self):
		'''
		Update values for phi, when phi is allowed to vary from cluster to cluster.
		'''
		l = self.pos['l']
		z = self.pos['z']
		
		l_star = np.unique(l)
		z_unmatched = np.delete(z,l_star)
		
		new_phi = np.zeros((self.n,self.n_dim,self.n_dim))

		nu_star = self.nu_phi + l.size            

		for i in range(self.n):
			if i in l_star:
				inds = (l == i)
				data_star = self.data[inds,...]
				z_star = z[i,...]
				nu_star = self.nu_phi + inds.size
				scale_star = self.cov_phi + np.sum((data_star - z_star)**2)
				
			else:
				scale_star = self.cov_phi
				nu_star = self.nu_phi
				
			new_phi[i,...] = stats.invwishart.rvs(df=nu_star,scale=scale_star)

		self.pos['phi'] = new_phi
        
	def draw_psi_loc(self):
		'''
		Update the location parameter of the hyperparameter distribution for z.
		
		Assume conjugacy, prior for psi_loc is just MVN.
		'''
		z = self.pos['z']
		l = self.pos['l']

		prior_mean_psi_loc = self.mean_psi_loc
		prior_cov_psi_loc = self.cov_psi_loc
		
		psi_cov = self.pos['psi_cov']
		psi_loc = self.pos['psi_loc']
		
		#if more than one dimension, need to compute matrix inverse of covariance matrices
		if self.n_dim > 1:
			psi_cov_inv = np.linalg.inv(psi_cov)
			prior_cov_psi_loc_inv = np.linalg.inv(prior_cov_psi_loc)
			
		else:
			psi_cov_inv = 1./psi_cov
			prior_cov_psi_loc_inv = 1./prior_cov_psi_loc

		l_star = np.unique(l)
		z_star = z[l_star,:]
		
		prec_star = prior_cov_psi_loc_inv + l_star.shape[0] * psi_cov_inv
		
		if self.n_dim > 1:
			cov_star = np.linalg.inv(prec_star)
		else:
			cov_star = 1./prec_star
			

		mean_star = np.dot(cov_star,(np.dot(prior_cov_psi_loc_inv,prior_mean_psi_loc) +\
			l_star.shape[0] * np.dot(psi_cov_inv,np.mean(z_star,axis=0))))           

		self.pos['psi_loc'] = stats.multivariate_normal.rvs(mean=mean_star,cov=prec_star)
		
		
    
	def draw_psi_cov(self):
		'''
		Update the scale parameter of the hyperparameter distribution for z.
		'''
		l = self.pos['l']
		z = self.pos['z']
		
		l_star = np.unique(l)
		z_star = z[l_star,:]
 		
		psi_loc = self.pos['psi_loc']

		if self.n_dim > 1:
			scale_inv = np.linalg.inv(self.cov_psi_cov)
		else:
			scale_inv = 1./self.cov_psi_cov

		nu_star = self.nu_psi_cov + l_star.size
		
		if self.n_dim == 1:
			scale_star = self.cov_psi_cov + np.sum((z_star[:,0] - psi_loc)**2)
		else:
			scale_star = self.cov_psi_cov + np.sum((z_star - psi_loc)**2)               

		#if self.n_dim > 1:
		self.pos['psi_cov'] = stats.invwishart.rvs(df=nu_star[0],scale=scale_star)
		
                                 
	def draw_alpha(self):
		'''
		Update the alpha parameter.
		'''
		v_l_star = self.pos['v_l_star']
		log_vals = np.log(1.-v_l_star)
		log_vals = log_vals[np.isfinite(log_vals)]
		log_p = np.sum(log_vals)

		self.pos['alpha'] = stats.gamma.rvs(self.a_alpha+self.pos['v_l_star'].size,
				scale=1./(self.b_alpha - log_p))
		
	def draw_post(self,grid=np.linspace(0.,1.,100),n_draw=1,n_burn=0,skip=0,calc_mean=True):
		"""
		Draw n samples from the posteior distribution
		"""
		mcmc_samples = self.p_arr.shape[0]
		req_samples = int((n_burn+n_draw)/(skip+1.))
				
		if req_samples > mcmc_samples:
			raise ValueError("You've requested more samples than are available in the MCMC output.")
			
		post_samples = n_draw
		
		#create the arrays to hold the posterior samples
		arr_shape = (grid.shape[0],post_samples)
		self.post_arr = np.zeros(arr_shape)
		
		p_samples = self.p_arr[n_burn::skip+1,:][:post_samples,:]
		z_samples = self.z_arr[:,:,n_burn::skip+1][:,:,:post_samples]
		if self.update_phi:
			phi_sample = self.phi_arr[...,n_burn::skip+1][...,:post_samples]
		else:
			phi_sample = np.dstack([self.pos['phi'] for i in range(n_draw)])
			
		#calculate the cumulative probability arrays so we can sample L
		cumu_p_samples = np.cumsum(p_samples,axis=1)
		
		#iterate over the grid array to evaluate p(y_0|data)
		for i in range(grid.shape[0]):	
			'''	
			#for each MCMC sample for the p-array, draw a value of L from it
			#this_l = np.argmin(np.ma.masked_less(cumu_p_samples \
			#	- np.random.uniform(size=post_samples).reshape(post_samples,1),0.),axis=1)
			
			#this_z = np.zeros(post_samples)
			#for j in range(post_samples):
			#	this_z[j] = z_samples[this_l[j],:,j]
			
			self.this_l = this_l
			self.this_z = this_z
			self.phi_sample = phi_sample
			
			#if phi has been sampled, need to select from it too
			if self.update_phi:
			 	phi_sample = self.phi_arr[:,this_l] 
			'''
			#At each point y_0, the functional is the weighted sum
			#of each of the individual kernels			
			if self.n_dim > 1:
				for j in range(p_samples.shape[0]):
					this_z = z_samples[:,:,j]	
					self.this_z = this_z
					self.post_arr[i,j] = np.sum(p_samples[j,:] * stats.multivariate_normal.pdf(grid[i,:],mean=this_z,cov=phi_sample[...,j]))
			else:
				for j in range(p_samples.shape[0]):
				    self.post_arr[i,j] = np.sum(p_samples[j,:] * stats.norm.pdf(grid[i],loc=z_samples[:,0,j],scale=np.sqrt(phi_sample[j])))
				#self.pdf = stats.norm.pdf(i,loc=this_z,scale=phi_sample)
				#self.post_arr[i,:] = stats.norm.pdf(grid[i],loc=this_z,scale=phi_sample)
			
		if calc_mean:
			self.mean_functional = np.mean(post_arr,axis=1)
			return self.post_arr,self.mean_functional
			
		return self.post_arr
			

class DPM_NIW(DPM):
	"""Class to implement non-parametric Bayesain inference for a multivariate normal - inverse wishart kernel.
	Expands DPM to allow covariance of kernel to also be updated.
	
	"""
	def __init__(self, arg):
		super(DPM, self).__init__()
		self.init_pos_phi(phi_init=phi_init)
		
		self.z_loc_arr = self.z_arr
		self.z_cov_arr = self.pos['phi']
		self.pos['phi'] = np.tile(self.pos['phi'],self.n).reshape(n,n_dim,n_dim)
		

		

            
        
        
    
    
    
    
    
    
