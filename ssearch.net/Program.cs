using Emgu.CV;
using Emgu.CV.XImgproc;
using Emgu.CV.Structure;

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Reflection;
using System.Configuration;

namespace ssearch.net
{
    class Program
    {
        static char mode = 'f';
        static int baseK = 150;
        static int incK = 150;
        static float sigma = 0.8f;
        static bool DoResize = true;
        static int newSize = 200;
        static void PrintUsage()
        {
            Console.WriteLine("Usage:");
            Console.WriteLine("{0} InputPath OutputPath [Mode(f|q)=f]", Assembly.GetExecutingAssembly().GetName().Name);
            Console.WriteLine("Mode: f=Fast, q=Quality");
            Console.WriteLine("InputPath and OutputPath must be both Directory or both File");
        }
        static void Main(string[] args)
        {
      
            if(args.Length < 2)
            {
                PrintUsage();
                return;
            }
            string inpPath = Path.GetFullPath(args[0]);
            string outpath = Path.GetFullPath(args[1]);

            bool inpIsDirectory;
            FileAttributes attr = File.GetAttributes(inpPath);
            inpIsDirectory = attr.HasFlag(FileAttributes.Directory);

       

            if (inpIsDirectory && !Directory.Exists(outpath))
            {
                try
                {
                    Directory.CreateDirectory(outpath);
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Can not create output directory: {0}", outpath);
                    Console.WriteLine(ex.Message);
                    return;
                }
            }



            if (args.Length > 2)
                mode = args[2].ToLower()[0];
            if(mode != 'f' && mode != 'q')
            {
                PrintUsage();
                return;
            }

            if(ConfigurationManager.AppSettings["baseK"]!= null)
            {
                int temp = 0;
                if (int.TryParse(ConfigurationManager.AppSettings["baseK"], out temp) && temp >= 0)
                    baseK = temp;
                else
                {
                    Console.WriteLine("The Value of baseK in Config File is Invalid");
                    return;
                }
            }
            if (ConfigurationManager.AppSettings["incK"] != null)
            {
                int temp = 0;
                if (int.TryParse(ConfigurationManager.AppSettings["incK"], out temp) && temp >= 0)
                    incK = temp;
                else
                {
                    Console.WriteLine("The Value of incK in Config File is Invalid");
                    return;
                }
            }
            if (ConfigurationManager.AppSettings["sigma"] != null)
            {
                float temp = 0;
                if (float.TryParse(ConfigurationManager.AppSettings["sigma"], out temp) && temp >= 0)
                    sigma = temp;
                else
                {
                    Console.WriteLine("The Value of sigma in Config File is Invalid");
                    return;
                }
            }
            if (ConfigurationManager.AppSettings["DoResize"] != null)
            {
                int temp = 0;
                if (int.TryParse(ConfigurationManager.AppSettings["DoResize"], out temp) && temp >= 0)
                    DoResize = (temp!=0);
                else
                {
                    Console.WriteLine("The Value of DoResize in Config File is Invalid");
                    return;
                }
            }
            if (ConfigurationManager.AppSettings["NewSize"] != null)
            {
                int temp = 0;
                if (int.TryParse(ConfigurationManager.AppSettings["NewSize"], out temp) && temp >= 0)
                    newSize = temp;
                else
                {
                    Console.WriteLine("The Value of NewSize in Config File is Invalid");
                    return;
                }
            }
            Console.WriteLine("Parameters:");
            Console.WriteLine($"\t baseK={baseK}");
            Console.WriteLine($"\t incK={incK}");
            Console.WriteLine($"\t sigma={sigma}");
            Console.WriteLine($"\t DoResize={DoResize}");
            Console.WriteLine($"\t NewSize={newSize}");
            CvInvoke.UseOptimized = true;
            CvInvoke.NumThreads = -1;

            if (!inpIsDirectory)
                ProcessFile(inpPath, outpath, true);
            else
            {
                int cnt = 0;
                foreach(string f in Directory.EnumerateFiles(inpPath, "*.jpg"))
                {
                    string outfile = Path.Combine(outpath, Path.GetFileNameWithoutExtension(f) + ".txt");
                    ProcessFile(f, outfile, false);
                    cnt++;
                    Console.Write("\r{0:D4}", cnt);
                }
            }
  
  
        }

        static void ProcessFile(string inpFile, string outFile, bool SaveOutImage)
        {
            SelectiveSearchSegmentation sss = new SelectiveSearchSegmentation();

            Mat im = CvInvoke.Imread(inpFile);
            Mat im2 ;
            float sfR = 1, sfC = 1;
            int newRows=im.Rows, newCols=im.Cols;
            if (DoResize)
            {

 
                if (im.Rows > im.Cols)
                {
                    newRows = newSize;
                    newCols = im.Cols * newRows / im.Rows;
                }
                else
                {
                    newCols = newSize;
                    newRows = im.Rows * newCols / im.Cols;
                }

                
            }
            float maxratio = 2.0f;
            if ((float)newRows / newCols > maxratio)
                newCols = (int)(newRows / maxratio);
            if ((float)newCols / newRows > maxratio)
                newRows = (int)(newCols / maxratio);


            if (newCols != im.Cols || newRows!=im.Rows)
            {
                im2 = new Mat();
                sfR = (float)im.Rows / newRows;
                sfC = (float)im.Cols / newCols;
                CvInvoke.Resize(im, im2, new Size(newCols, newRows));
            }
            else
                im2 = im;

            sss.SetBaseImage(im2);

            if (mode == 'f')
                sss.SwitchToSelectiveSearchFast(baseK, incK, sigma);
            else
                sss.SwitchToSelectiveSearchQuality(baseK, incK, sigma);


            Rectangle[] rects = new Rectangle[] { };
            try
            {
               rects = sss.Process();
            }
            catch(Exception ex)
            {
                Console.WriteLine("error: " + ex.Message);
                return;
            }

            int N = rects.Length;
            for (int i = 0; i < N; i++)
            {
                rects[i].X = (int)(rects[i].X * sfC);
                rects[i].Y = (int)(rects[i].Y * sfR);
                rects[i].Width = (int)(rects[i].Width * sfC);
                rects[i].Height = (int)(rects[i].Height * sfR);
            }

            StreamWriter sw = new StreamWriter(outFile);
            for (int i = 0; i < N; i++)
                sw.WriteLine("{0}, {1}, {2}, {3}", rects[i].X, rects[i].Y, rects[i].Width, rects[i].Height);
            sw.Close();

            if (SaveOutImage)
            {
                int nshow = 100;
                if (nshow > N)
                    nshow = N;

                Mat imout = im.Clone();
                for (int i = 0; i < nshow; i++)
                {
                    CvInvoke.Rectangle(imout, rects[i], new MCvScalar(0, 255, 0));
                }
                CvInvoke.Imwrite("out.jpg", imout);
            }
        }
    }
}
