from main import TechStack

us_soccer_stack = TechStack(
    primary_previous_solution=OrchestrationTool.HOME_GROWN_BASIC,
    secondary_previous_solutions=[OrchestrationTool.AWS_LAMBDA_FUNCTIONS],
    cloud_provider="AWS",
)

training_data = {"006Rm00000QuHC6IAN": us_soccer_stack}
