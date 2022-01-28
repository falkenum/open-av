from http import server

class Handler(server.BaseHTTPRequestHandler):
    def do_POST(self):
        print(f"data: {self.rfile.readline()}")
        self.send_response(server.http.HTTPStatus.OK)
        self.end_headers()

    def do_GET(self):
        self.send_response(server.http.HTTPStatus.OK)
        self.end_headers()

s = server.HTTPServer(("0.0.0.0", 8080), Handler)

s.serve_forever()
