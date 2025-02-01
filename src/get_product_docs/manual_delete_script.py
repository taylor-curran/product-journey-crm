import turbopuffer as tpuf


def delete_namespace(namespace: str):
    """Delete a TurboPuffer namespace"""
    print(f"Deleting namespace: {namespace}")

    try:
        ns = tpuf.Namespace(namespace)
        ns.delete_all()  # Using delete_all() as per the docs
        print(f"Successfully deleted namespace: {namespace}")
    except tpuf.APIError as e:  # Specifically catch APIError
        print(f"Error deleting namespace {namespace}: {e}")


if __name__ == "__main__":
    # Delete the test-tay namespace
    delete_namespace("tay-test")
