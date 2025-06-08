import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Mail, MapPin, GraduationCap } from "lucide-react"

export function Authors() {
  return (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle>Research Authors</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <GraduationCap className="h-5 w-5 text-blue-600" />
              <h3 className="font-semibold text-lg">Victor Baraka M Muiruri</h3>
              
            </div>
            <div className="flex items-center space-x-2 text-gray-600">
              <Mail className="h-4 w-4" />
              <span className="text-sm">victormuiruri007@gmail.com</span>
            </div>
            <div className="flex items-center space-x-2 text-gray-600">
              <MapPin className="h-4 w-4" />
              <span className="text-sm">Department of Electrical and Electronics Engineering</span>
            </div>
            <p className="text-sm text-gray-600">ORCID: 0009-0006-6287-0097</p>
          </div>

          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <GraduationCap className="h-5 w-5 text-blue-600" />
              <h3 className="font-semibold text-lg">Dr. Davies Rene Segera</h3>
              <Badge variant="outline">Corresponding Author</Badge>
            </div>
            <div className="flex items-center space-x-2 text-gray-600">
              <Mail className="h-4 w-4" />
              <span className="text-sm">davies.segera@uonbi.ac.ke</span>
            </div>
            <div className="flex items-center space-x-2 text-gray-600">
              <MapPin className="h-4 w-4" />
              <span className="text-sm">Department of Electrical and Electronics Engineering</span>
            </div>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t">
          <div className="flex items-center space-x-2 text-gray-600">
            <MapPin className="h-4 w-4" />
            <span className="font-medium">University of Nairobi, Nairobi, Kenya</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
