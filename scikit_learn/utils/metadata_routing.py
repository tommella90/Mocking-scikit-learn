#%%
class SimplifiedParameterRouter:
    def route_params(self, params, method):
        """
        Simplified parameter routing.

        Parameters:
        - params (dict): A dictionary of all provided metadata/parameters.
        - method (str): The name of the method for which parameters are requested and routed.

        Returns:
        - routed_params (dict): A dictionary of parameters that are relevant to the specified method.
        """
        routed_params = {}
        prefix = f"{method}__"

        for param, value in params.items():
            if param.startswith(prefix):
                # Extract the parameter name without the method prefix
                param_name = param[len(prefix):]
                routed_params[param_name] = value

        return routed_params
    
def process_routing(self, obj, method, **kwargs):
    routed_params = self.route_params()
    return routed_params


#%%
router = SimplifiedParameterRouter()
params = {
    "transform__param1": "value1",
    "transform__param2": "value2",
    "estimator__param3": "value3"
}
method = "transform"

routed_params = router.route_params(params, method)
routed_params
#
# %%

for param, value in params.items():
    print(param, value)
# %%
